#include "aw_test_utils/AmrexTest.H"
#include "aw_test_utils/MeshTest.H"

namespace amr_wind_tests {

namespace analytical_function {

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real phi_eval(
    int degree,
    const amrex::Real* coeff,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    amrex::Real phi = 0.0;

    int ind = 0.0;
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k <= degree; ++k) {
                phi += coeff[ind++] * std::pow(x, i) * std::pow(y, j) *
                       std::pow(z, k);
            }
        }
    }

    return phi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real dphidx_eval(
    int degree,
    const amrex::Real* coeff,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    amrex::Real dphidx = 0.0;

    int ind = 0.0;
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k <= degree; ++k) {
                dphidx += i * coeff[ind++] * std::pow(x, amrex::max(i - 1, 0)) *
                          std::pow(y, j) * std::pow(z, k);
            }
        }
    }

    return dphidx;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real dphidy_eval(
    int degree,
    const amrex::Real* coeff,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    amrex::Real dphidy = 0.0;

    int ind = 0.0;
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k <= degree; ++k) {
                dphidy += j * coeff[ind++] * std::pow(x, i) *
                          std::pow(y, amrex::max(j - 1, 0)) * std::pow(z, k);
            }
        }
    }

    return dphidy;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real dphidz_eval(
    int degree,
    const amrex::Real* coeff,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    amrex::Real dphidz = 0.0;

    int ind = 0.0;
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k <= degree; ++k) {
                dphidz += k * coeff[ind++] * std::pow(x, i) * std::pow(y, j) *
                          std::pow(z, amrex::max(k - 1, 0));
            }
        }
    }

    return dphidz;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real laplacian(
    int degree,
    const amrex::Real* coeffx,
    const amrex::Real* coeffy,
    const amrex::Real* coeffz,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    amrex::Real lap = 0.0;

    int ind = 0;
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k <= degree; ++k) {
                lap += i * amrex::max(i - 1, 0) * coeffx[ind] *
                       std::pow(x, amrex::max(i - 2, 0)) * std::pow(y, j) *
                       std::pow(z, k);
                lap += j * amrex::max(j - 1, 0) * coeffy[ind] * std::pow(x, i) *
                       std::pow(y, amrex::max(j - 2, 0)) * std::pow(z, k);
                lap += k * amrex::max(k - 1, 0) * coeffz[ind] * std::pow(x, i) *
                       std::pow(y, j) * std::pow(z, amrex::max(k - 2, 0));

                ++ind;
            }
        }
    }

    return lap;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real divergence(
    int degree,
    const amrex::Real* coeffx,
    const amrex::Real* coeffy,
    const amrex::Real* coeffz,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    amrex::Real div = 0.0;

    int ind = 0;
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k <= degree; ++k) {
                div += i * coeffx[ind] * std::pow(x, amrex::max(i - 1, 0)) *
                       std::pow(y, j) * std::pow(z, k);
                div += j * coeffy[ind] * std::pow(x, i) *
                       std::pow(y, amrex::max(j - 1, 0)) * std::pow(z, k);
                div += k * coeffz[ind] * std::pow(x, i) * std::pow(y, j) *
                       std::pow(z, amrex::max(k - 1, 0));

                ++ind;
            }
        }
    }

    return div;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real strainrate(
    int degree,
    const amrex::Real* coeffx,
    const amrex::Real* coeffy,
    const amrex::Real* coeffz,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    const amrex::Real ux = dphidx_eval(degree, coeffx, x, y, z);
    const amrex::Real uy = dphidy_eval(degree, coeffx, x, y, z);
    const amrex::Real uz = dphidz_eval(degree, coeffx, x, y, z);

    const amrex::Real vx = dphidx_eval(degree, coeffy, x, y, z);
    const amrex::Real vy = dphidy_eval(degree, coeffy, x, y, z);
    const amrex::Real vz = dphidz_eval(degree, coeffy, x, y, z);

    const amrex::Real wx = dphidx_eval(degree, coeffz, x, y, z);
    const amrex::Real wy = dphidy_eval(degree, coeffz, x, y, z);
    const amrex::Real wz = dphidz_eval(degree, coeffz, x, y, z);

    return std::sqrt(
        2.0 * ux * ux + 2.0 * vy * vy + 2.0 * wz * wz + (uy + vx) * (uy + vx) +
        (vz + wy) * (vz + wy) + (wx + uz) * (wx + uz));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real curvature(
    int degree,
    const amrex::Real* coeff,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z) noexcept
{

    const amrex::Real phix = dphidx_eval(degree, coeff, x, y, z);
    const amrex::Real phiy = dphidy_eval(degree, coeff, x, y, z);
    const amrex::Real phiz = dphidz_eval(degree, coeff, x, y, z);

    amrex::Real phixx = 0.0;
    amrex::Real phixy = 0.0;
    amrex::Real phiyy = 0.0;
    amrex::Real phixz = 0.0;
    amrex::Real phiyz = 0.0;
    amrex::Real phizz = 0.0;

    int ind = 0;
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k <= degree; ++k) {

                phixx += i * amrex::max(i - 1, 0) * coeff[ind] *
                         std::pow(x, amrex::max(i - 2, 0)) * std::pow(y, j) *
                         std::pow(z, k);

                phixy += i * j * coeff[ind] *
                         std::pow(x, amrex::max(i - 1, 0)) *
                         std::pow(y, amrex::max(j - 1, 0)) * std::pow(z, k);

                phiyy += j * amrex::max(j - 1, 0) * coeff[ind] *
                         std::pow(x, i) * std::pow(y, amrex::max(j - 2, 0)) *
                         std::pow(z, k);

                phixz += i * k * coeff[ind] *
                         std::pow(x, amrex::max(i - 1, 0)) * std::pow(y, j) *
                         std::pow(z, amrex::max(k - 1, 0));

                phiyz += j * k * coeff[ind] * std::pow(x, i) *
                         std::pow(y, amrex::max(j - 1, 0)) *
                         std::pow(z, amrex::max(k - 1, 0));

                phizz += k * amrex::max(k - 1, 0) * coeff[ind] *
                         std::pow(x, i) * std::pow(y, j) *
                         std::pow(z, amrex::max(k - 2, 0));

                ++ind;
            }
        }
    }

    return -(phix * phix * phiyy - 2. * phix * phiy * phixy +
             phiy * phiy * phixx + phix * phix * phizz -
             2. * phix * phiz * phixz + phiz * phiz * phixx +
             phiy * phiy * phizz - 2 * phiy * phiz * phiyz +
             phiz * phiz * phiyy) /
           std::pow(phix * phix + phiy * phiy + phiz * phiz, 1.5);
}

} // namespace analytical_function

} // namespace amr_wind_tests