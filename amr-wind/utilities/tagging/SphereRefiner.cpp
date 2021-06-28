#include "amr-wind/utilities/tagging/SphereRefiner.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace tagging {

SphereRefiner::SphereRefiner(const CFDSim&, const std::string& key)
{
    amrex::ParmParse pp(key);

    amrex::Vector<amrex::Real> tmp_vec;

    // Center of the sphere
    pp.getarr("center", tmp_vec);
    AMREX_ALWAYS_ASSERT(tmp_vec.size() == AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) m_center[i] = tmp_vec[i];

    // Radial extent of the sphere, always read in from input file
    pp.get("radius", m_radius);
}

void SphereRefiner::operator()(
    const amrex::Box& bx,
    const amrex::Geometry& geom,
    const amrex::Array4<amrex::TagBox::TagType>& tag) const
{
    const auto center = m_center;
    const auto rad2 = m_radius * m_radius;
    const auto& problo = geom.ProbLoArray();
    const auto& dx = geom.CellSizeArray();

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
        const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];

        // Position vector of the cell center
        const vs::Vector pt(x, y, z);
        // Vector relative to the center of the sphere
        const vs::Vector pvec = pt - center;

        // Check if the point lies in the sphere
        if ((pvec & pvec) <= rad2) {
            tag(i, j, k) = amrex::TagBox::SET;
        }
    });
}

} // namespace tagging
} // namespace amr_wind
