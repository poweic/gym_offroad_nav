// cython methods to speed-up evaluation
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.14159265359
#define RAD2DEG (180. / PI)
#define M2CM 100.

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

void mask(uint8_t* images, uint32_t batch_size, uint32_t height, uint32_t width,
    uint8_t threshold, uint32_t random_seed) {

  // If seed is set to 1, the generator is reinitialized to its initial value
  // and produces the same values as before any call to rand or srand.
  // Make sure it's not 1, and set the random_seed
  assert(random_seed > 1);
  srand(random_seed);
  gThreshold = threshold;

  uint32_t step = width*height;
  for (uint32_t k=0; k<batch_size; ++k)
    mask_single_image(images + k*step, height, width);
}
