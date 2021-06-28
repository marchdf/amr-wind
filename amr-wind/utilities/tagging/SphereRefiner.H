#ifndef SPHEREREFINER_H
#define SPHEREREFINER_H

#include "amr-wind/utilities/tagging/GeometryRefinement.H"
#include "amr-wind/core/vs/vector_space.H"

namespace amr_wind {
namespace tagging {

/** Refine region inside a sphere
 *  \ingroup amr_utils
 *
 *  The sphere is centered on `center` and the `radius` option
 *  specifies the radius.
 */
class SphereRefiner : public GeometryType::Register<SphereRefiner>
{
public:
    static std::string identifier() { return "sphere"; }

    SphereRefiner(const CFDSim&, const std::string& key);

    virtual ~SphereRefiner() = default;

    void operator()(
        const amrex::Box&,
        const amrex::Geometry& geom,
        const amrex::Array4<amrex::TagBox::TagType>& tags) const override;

private:
    vs::Vector m_center;

    amrex::Real m_radius;
};

} // namespace tagging
} // namespace amr_wind

#endif /* SPHEREREFINER_H */
