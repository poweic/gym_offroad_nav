// cython methods to speed-up evaluation
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "opencv2/opencv.hpp"
#define PI 3.14159265359
#define RAD2DEG (180. / PI)
#define M2CM 100.

using namespace cv;
using namespace std;

#define GET_VALUE(i, j) (image[(i)*height+(j)])
// threshold
uint8_t gThreshold;

// Provide the compiler with branch prediction information, which may gain huge
// speed-up when you're pretty sure a conditional statement is almost always
// true or almost always false. See the following refs.
// [1] http://stackoverflow.com/questions/7346929/why-do-we-use-builtin-expect-when-a-straightforward-way-is-to-use-if-else
// [2] http://blog.man7.org/2012/10/how-much-do-builtinexpect-likely-and.html
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

// absolute value of x
template <typename T>
inline T abs_(T x) { return (x >= 0) ? x : -x; }

// sign of x, return either -1, 0, or 1
template <typename T>
inline T sign_(T x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }

// max of x and y
template <typename T>
inline T max_(T x, T y) { return (x > y) ? x : y; }

// max of x and y
template <typename T>
inline T min_(T x, T y) { return (x < y) ? x : y; }

// Use C macro ##, which is a token-pasting operator
#define trace_one_step(x, a) { \
  if (unlikely(dec##x >= 0)) { \
    dec##x -= a; \
    x += s##x; \
  } \
  dec##x += a##x; \
}

template <int axis>
inline void __attribute__((always_inline)) trace_along_axis(
    uint8_t* const &image,
    const int32_t &m0, const int32_t &m1, const int32_t &g1p,
    int32_t &x,  int32_t &y,
    int32_t &dx, int32_t &dy) {

    int32_t sx = sign_(dx);
    int32_t sy = sign_(dy);

    dx = abs_(dx);
    dy = abs_(dy);

    int32_t ax = dx << 1;
    int32_t ay = dy << 1;

    const int32_t L = max_(dx, dy);
    int32_t a = L << 1;

    int32_t decx = ax - L;
    int32_t decy = ay - L;

    bool collided = false;

    for (int32_t k=0; k<L+1; ++k) {
      unsigned int xx = x & m0;
      unsigned int yy = y & m1;
      unsigned int idx = (xx << g1p) + yy;

      uint8_t& value = image[idx];

      // Only obstacles, bushes, and trees has value < 0
      if (value < gThreshold) {
	// stop_prob is the absolute value, but since gThreshold < 0, value
	// is always smaller than 0
	uint8_t stop_prob = 128 - value;
	uint8_t sample = rand() & 0x77;

	if (sample < stop_prob)
	  collided = true;
      }

      if (collided)
	value = 0;

      if (axis == 0) x += sx; else trace_one_step(x, a);
      if (axis == 1) y += sy; else trace_one_step(y, a);
    }
}

// x is the 1st dimension (i.e. shape[0], or height)
// y is the 2nd dimension (i.e. shape[1], or width )
void bresenham_trace(
    int32_t x1, int32_t y1,
    int32_t x2, int32_t y2,
    uint8_t* image, const uint32_t g0, const uint32_t g1) {

    int32_t dx = x2 - x1;
    int32_t dy = y2 - y1;

    int axis_idx = abs_(dx) < abs_(dy);

    int32_t m0 = g0 - 1;
    int32_t m1 = g1 - 1;
    int32_t g1p = log2(g1);

    if (axis_idx == 0)
      trace_along_axis<0>(image, m0, m1, g1p, x1, y1, dx, dy);
    else
      trace_along_axis<1>(image, m0, m1, g1p, x1, y1, dx, dy);
}

void mask_single_image(uint8_t* image, uint32_t height, uint32_t width) {

  for (uint32_t j=0; j<height; ++j) {
    // Left boundary
    bresenham_trace(height - 1, width/2 - 1, j, 0, image, width, height);

    // Right boundary
    bresenham_trace(height - 1, width/2    , j, width-1, image, width, height);
  }

  // Left part of Top boundary
  for (uint32_t i=0; i<width/2; ++i)
    bresenham_trace(height - 2, width/2 - 1, 0, i, image, width, height);

  // Right part of Top boundary
  for (uint32_t i=width/2; i<width; ++i)
    bresenham_trace(height - 2, width/2    , 0, i, image, width, height);
}

void rotated_rect(
    cv::Mat& dst, const cv::Mat& src, Point2f& pivot,
    const int32_t* const centers, const float& angle, const float& scale) {

  cv::Mat M = cv::getRotationMatrix2D(pivot, angle, scale);
  M.at<double>(0, 2) += centers[0] - pivot.x;
  M.at<double>(1, 2) += centers[1] - pivot.y;

  cv::warpAffine(src, dst, M, dst.size(),
      cv::WARP_INVERSE_MAP + cv::INTER_NEAREST,
      cv::BORDER_REPLICATE);

  // cout << M << endl;
}

void draw_objects(
    cv::Mat& dst,
    const int32_t& pivot_x, const int32_t& pivot_y, const float& scale,
    const double* const obj_positions, const uint8_t* const valids, uint32_t n_obj,
    const double* const state, float cell_size, uint32_t radius) {

  const double& x0 = state[0];
  const double& y0 = state[1];
  const double& theta = state[2];

  double c = cos(theta);
  double s = sin(theta);

  for (uint32_t i=0; i<n_obj; ++i) {
    const auto obj_pos = obj_positions + i*2;
    bool valid = valids[i];

    if (not valid)
      continue;

    // Compute difference
    double dx_ = (obj_pos[0] - x0) / scale;
    double dy_ = (obj_pos[1] - y0) / scale;

    // Rotate
    double dx =  c * dx_ + s * dy_;
    double dy = -s * dx_ + c * dy_;

    // convert to grid index
    int32_t ix = dx / cell_size;
    int32_t iy = dy / cell_size;

    ix =  ix + pivot_x;
    iy = -iy + pivot_y;

    // draw object as a circle
    cv::circle(dst, Point(ix, iy), radius, Scalar(255, 255, 255), -1);
  }
}

void mask(
    uint8_t* images,
    uint8_t* reward_map,
    const int32_t* const centers,
    const float* const angles,
    const int32_t& pivot_x, const int32_t& pivot_y, const float& scale,
    const uint32_t& batch_size, const uint32_t& height, const uint32_t& width,
    const uint32_t& in_height, const uint32_t& in_width,
    const double* const obj_positions, const uint8_t* const valids, uint32_t n_obj,
    const double* const states, float cell_size, uint32_t radius,
    const uint8_t& threshold, const uint32_t& random_seed) {

  uint32_t step = width*height;

  // If seed is set to 1, the generator is reinitialized to its initial value
  // and produces the same values as before any call to rand or srand.
  // Make sure it's not 1, and set the random_seed
  assert(random_seed > 1);
  srand(random_seed);
  gThreshold = threshold;

  // create src mat (wrapper) from reward_map
  Point2f pivot(pivot_x, pivot_y);
  cv::Mat src(in_height, in_width, CV_8UC1, (void*) reward_map);

  for (uint32_t i=0; i<batch_size; ++i) {

    uint8_t* image = images + i*step;
    cv::Mat dst(height, width, CV_8UC1, (void*) image);

    // rotate
    rotated_rect(dst, src, pivot, centers + i*2, angles[i], scale);

    // ray-casting
    mask_single_image(image, height, width);

    // draw objects on image
    draw_objects(dst, pivot_x, pivot_y, scale,
	obj_positions, valids + i*n_obj, n_obj, states + i*6, cell_size, radius);
  }
}
