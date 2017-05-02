// cython methods to speed-up evaluation
#include <stdint.h>
#include <math.h>
#define PI 3.14159265359
#define RAD2DEG (180. / PI)
#define M2CM 100.

#define GET_VALUE(i, j) (image[(i)*height+(j)])
// threshold
const float kThreshold = -0.9;

// Provide the compiler with branch prediction information, which may gain huge
// speed-up when you're pretty sure a conditional statement is almost always
// true or almost always false. See the following refs.
// [1] http://stackoverflow.com/questions/7346929/why-do-we-use-builtin-expect-when-a-straightforward-way-is-to-use-if-else
// [2] http://blog.man7.org/2012/10/how-much-do-builtinexpect-likely-and.html
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

// absolute value of x
inline int abs_(int x) { return (x >= 0) ? x : -x; }

// sign of x, return either -1, 0, or 1
inline int sign_(int x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }

// max of x and y
inline int max_(int x, int y) { return (x > y) ? x : y; }

// max of x and y
inline int min_(int x, int y) { return (x < y) ? x : y; }

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
    float* const &image,
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

      float& value = image[idx];

      if (image[idx] < kThreshold)
	collided = true;
      
      if (collided)
	value = -1;

      if (axis == 0) x += sx; else trace_one_step(x, a);
      if (axis == 1) y += sy; else trace_one_step(y, a);
    }
}

// x is the 1st dimension (i.e. shape[0], or height)
// y is the 2nd dimension (i.e. shape[1], or width )
void bresenham_trace(
    int32_t x1, int32_t y1,
    int32_t x2, int32_t y2,
    float* image, const uint32_t g0, const uint32_t g1) {

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

void mask_single_image(float* image, uint32_t height, uint32_t width) {

  int32_t x1 = height - 1, y1 = width/2 - 1;

  uint32_t step = 1;
  for (uint32_t j=0; j<height; j+=step) {
    bresenham_trace(x1, y1, j, 0, image, width, height);
    bresenham_trace(x1, y1, j, width-1, image, width, height);
  }

  for (uint32_t i=0; i<width; i+=step)
    bresenham_trace(x1, y1, 0, i, image, width, height);
}

void mask(float* images, uint32_t batch_size, uint32_t height, uint32_t width) {
  for (uint32_t k=0; k<batch_size; ++k) {
    float* image = images + k*width*height;
    mask_single_image(image, height, width);
  }
}

/*
#define T_VISITED 100
#define NT_VISITED -100
#define BOUNDARY -50

// run bfs recursively and return whether (i, j) is traversable
bool bfs_inner(float* image, uint32_t height, uint32_t width, uint32_t i, uint32_t j) {

  float& value = GET_VALUE(i, j);

  // For those visited, store +100 for traversable, -100 for non-traversable
  // Test if visited, if yes, continue
  if (value == T_VISITED or value == NT_VISITED)
    return value == T_VISITED;

  if (value == -50)
    return false;

  // if not visited, and it's traversable, then store +100
  if (value > kThreshold) {
    value = +T_VISITED;
    return true;
  }

  // if not visted and non-traversable, run bfs_inner recursively
  value = NT_VISITED;
  bool l = (i == 0)        ? false : bfs_inner(image, height, width, i-1, j);
  bool r = (i == width-1)  ? false : bfs_inner(image, height, width, i+1, j);
  bool t = (j == 0)        ? false : bfs_inner(image, height, width, i, j-1);
  bool b = (j == height-1) ? false : bfs_inner(image, height, width, i, j+1);

  if (l or r or t or b)
    value = BOUNDARY;

  return false;
}

void bfs(float* image, uint32_t height, uint32_t width) {
  for (uint32_t i=0; i<width; ++i) {
    for (uint32_t j=0; j<height; ++j) {
      bfs_inner(image, height, width, i, j);
    }
  }

  for (uint32_t i=0; i<width; ++i) {
    for (uint32_t j=0; j<height; ++j) {
      float& value = image[i*height+j];
      value = (value == BOUNDARY);
    }
  }
}
*/
