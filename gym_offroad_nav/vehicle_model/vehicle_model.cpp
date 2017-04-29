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

void step(
    double* x, double* s, const double* u, const double* n,
    uint32_t n_sub_steps, uint32_t batch_size,
    float dt, float noise_level, float wheelbase, float drift) {

  // syntax sugar
  uint32_t& b = batch_size;

  for (uint32_t i=0; i<n_sub_steps; ++i) {
    for (uint32_t j=0; j<b; ++j) {
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

      double dx = (c * s3 - s * s4) * dt;
      double dy = (s * s3 + c * s4) * dt;
      double dw = s5 * dt;

      s0 += dx * (1. + n0 * noise_level);
      s1 += dy * (1. + n1 * noise_level);
      s2 += dw * (1. + n2 * noise_level);
      s3 = y0;
      s4 = y1;
      s5 = y2;
    }
  }

}
