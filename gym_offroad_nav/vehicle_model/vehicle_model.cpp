// cython methods to speed-up evaluation
#include <stdint.h>
#include <math.h>
#define PI 3.14159265359
#define RAD2DEG (180. / PI)
#define M2CM 100.

/*
   |              ( A )                              ( B )
   | x0           x1        x2        x3                u0   u1
-- | -------------------------------------------------  --------
x0 | 0.9933   (*) 0.        0.        0.                0    1
x1 | 0.           0.9519    0.        0.                1    0
x2 | 0.           0.        1.902    -0.9069            0    1
x3 | 0.           0.        1.        0.                0    0

   |              ( C )                              ( D )
   | x0           x1          x2           x3           u0   u1
-- | -------------------------------------------------  --------
y0 | 0.01740  (*) 0.          0.           0.           0    -2.594 (*)
y1 | 0.           0.04152     0.           0.           0    0
y2 | 0.           0.          -0.002347    -0.002272    0    0
*/

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}
#define STEER_TO_YAWRATE(vf, steer, wheelbase) (vf * tan(steer) / wheelbase)

#define GET_REWARD(ix, iy) (reward_map[(y_max - 1 - (iy)) * width + ((ix) - x_min)])

#define VEC_NORM(x, y) (sqrt((x)*(x) + (y)*(y)))

inline float bilinear_reward_lookup(const double &x, const double &y,
    const float* const reward_map, uint32_t height, uint32_t width,
    int32_t x_min, int32_t x_max, int32_t y_min, int32_t y_max, float cell_size) {

  int32_t ix = floor(x / cell_size);
  int32_t iy = floor(y / cell_size);

  double xx = x / cell_size - ix;
  double yy = y / cell_size - iy;

  // printf("ix = %d, iy = %d\n", ix, iy);

  float r00 = GET_REWARD(ix    , iy    );
  float r01 = GET_REWARD(ix    , iy + 1);
  float r10 = GET_REWARD(ix + 1, iy    );
  float r11 = GET_REWARD(ix + 1, iy + 1);

  float w00 = (1.-xx) * (1.-yy);
  float w01 = (   yy) * (1.-xx);
  float w10 = (   xx) * (1.-yy);
  float w11 = (   xx) * (   yy);

  float r = r00*w00 + r01*w01 + r10*w10 + r11*w11;

  return r;
}

void step(
    double* x, double* s, const double* u, const double* n,
    uint32_t n_sub_steps, uint32_t batch_size,
    float dt, float noise_level, float wheelbase, float drift,
    float* rewards, const float* const reward_map, uint32_t height, uint32_t width,
    int32_t x_min, int32_t x_max, int32_t y_min, int32_t y_max, float cell_size) {

  // syntax sugar
  uint32_t& b = batch_size;

  for (uint32_t j=0; j<b; ++j) {
    double prev_vx, prev_vy;

    for (uint32_t i=0; i<n_sub_steps; ++i) {
      double& s0 = s[j + 0*b];
      double& s1 = s[j + 1*b];
      double& s2 = s[j + 2*b];
      double& s3 = s[j + 3*b];
      double& s4 = s[j + 4*b];
      double& s5 = s[j + 5*b];

      double& x0 = x[j + 0*b];
      double& x1 = x[j + 1*b];
      double& x2 = x[j + 2*b];
      double& x3 = x[j + 3*b];

      const double& u0 = u[j + 0*b];  // forward velocity command
      const double& u1 = u[j + 1*b];  // steering angle command
      double u_yawrate = STEER_TO_YAWRATE(s4, u1, wheelbase);

      const double& n0 = n[(i*3 + 0)*b + j];
      const double& n1 = n[(i*3 + 1)*b + j];
      const double& n2 = n[(i*3 + 2)*b + j];

      // printf("x = [%f, %f, %f, %f]\n", x0, x1, x2, x3);
      // printf("s = [%f, %f, %f, %f, %f, %f]\n", s0, s1, s2, s3, s4, s5);
      // printf("u = [%f, %f]\n", u0, u1);

      double y0 = (0.01740 / M2CM) * x0 * drift - (2.594 * RAD2DEG / M2CM ) * u_yawrate * drift;
      double y1 = (0.04152 / M2CM) * x1;
      double y2 = (-0.002347 / RAD2DEG) * x2 - (0.002272 / RAD2DEG) * x3;

      double x0_ = 0.9933 * x0 * drift + RAD2DEG * u_yawrate;
      double x1_ = 0.9519 * x1 + M2CM * u0;
      double x2_ = 1.902 * x2 - 0.9069 * x3 + RAD2DEG * u_yawrate;
      double x3_ = x2;

      x0 = x0_;
      x1 = x1_;
      x2 = x2_;
      x3 = x3_;

      double c = cos(s2);
      double s = sin(s2);

      // Use uni-cycle model (actually just rotation matrix) to compute vehicle
      // velocity in static global frame. This will later be used to compute
      // displacement/acceleration in static global frame.
      double vx = (c * s3 - s * s4);
      double vy = (s * s3 + c * s4);

      // Formula: dx = v dt (add additional environmental noise)
      double dx = vx * (1. + n0 * noise_level) * dt;
      double dy = vy * (1. + n1 * noise_level) * dt;
      double dw = s5 * (1. + n2 * noise_level) * dt;

      // compute the displacement (ds = sqrt(dx^2 + dy^2)) along the path, the
      // longer the distance travel on "smooth trail", the higher the reward.
      float displacement = VEC_NORM(dx, dy);
      float r = bilinear_reward_lookup(s0 + dx / 2, s1 + dy / 2,
	  reward_map, height, width, x_min, x_max, y_min, y_max, cell_size);

      rewards[j] += displacement * r;

      float v = VEC_NORM(vx, vy);
      rewards[j] -= 0.01 * exp(-3 * v);

      // Skip i == 0, penalize large acceleration
      // Formula: a = dv / dt
      if (i > 0) {
	float dvx = vx - prev_vx;
	float dvy = vy - prev_vy;
	float dv = VEC_NORM(dvx, dvy);
	float acc = dv / dt;
	rewards[j] -= 0.01 * acc;
      }

      s0 += dx;
      s1 += dy;
      s2 += dw;
      s3 = y0;
      s4 = y1;
      s5 = y2;

      prev_vx = vx;
      prev_vy = vy;
    }
  }

  /* for testing (passed)
  for (uint32_t i=1; i<height - 1; ++i) {
    for (uint32_t j=1; j<width - 1; ++j) {
      float x = (double) (float(j) + float(x_min)) * cell_size;
      float y = (double) (float(y_max) - 1 - float(i)) * cell_size;
      float r = bilinear_reward_lookup(x, y, reward_map, height, width,
	  x_min, x_max, y_min, y_max, cell_size);
      rewards[i * width + j] = r;
    }
  }
  */
}
