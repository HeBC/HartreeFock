#include <Python.h>

#include "HartreeFock.h"
#include "AngMom.h"
#include "GCM_Tools.h"
#include <string>
#include <sstream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "HFHybridBindings.h"


namespace py = pybind11;

PYBIND11_MODULE(pyHFAndHFB, m)
{
    bind_hybrid_backend(m);
    m.def("hybrid_cg", [](int j1,int m1,int j2,int m2,int j,int mm) {
        if(j1<0 || j2<0 || j<0 || j1>100 || j2>100 || j>100)
            throw std::invalid_argument("invalid doubled angular momentum");
        if(std::abs(m1)>j1 || std::abs(m2)>j2 || std::abs(mm)>j ||
           (j1-m1)%2 || (j2-m2)%2 || (j-mm)%2 || m1+m2!=mm ||
           j>j1+j2 || j<std::abs(j1-j2) || (j1+j2+j)%2) return 0.;
        return AngMom::cgc(.5*j,.5*mm,.5*j1,.5*m1,.5*j2,.5*m2);
    }, "Clebsch-Gordan coefficient; every argument is doubled.");
    m.doc() = "Python bindings for HF code";

    py::class_<ModelSpace>(m, "ModelSpace")
        .def(py::init<>())
        .def("Set_RefString", &ModelSpace::Set_RefString)
        .def("InitialModelSpace_HF", &ModelSpace::InitialModelSpace_HF)
        .def("SetProtonNum", &ModelSpace::SetProtonNum)
        .def("SetNeutronNum", &ModelSpace::SetNeutronNum)
        .def("SetShapeConstrained", &ModelSpace::SetShapeConstrained)
        .def("SetShapeQ", &ModelSpace::SetShapeQ)
        .def("Set_Jz_constraint", &ModelSpace::Set_Jz_constraint)
        .def("SetTargetJz", &ModelSpace::SetTargetJz)
        .def("Set_Jx_constraint", &ModelSpace::Set_Jx_constraint)
        .def("SetTargetJx", &ModelSpace::SetTargetJx)
        .def("PrintAllParameters_HF", &ModelSpace::PrintAllParameters_HF)
        .def("Set_hw", &ModelSpace::Set_hw)
        .def("Get_hw", &ModelSpace::Get_hw)
        .def("Set_MeshType", &ModelSpace::Set_MeshType)
        .def("SetAMProjected_JMK", &ModelSpace::SetAMProjected_JMK)
        .def("SetProjected_parity", &ModelSpace::SetProjected_parity)
        .def("SetGuassQuadMesh", &ModelSpace::SetGuassQuadMesh);

        
    py::class_<ReadWriteFiles>(m, "ReadWriteFiles")
        .def(py::init<>())
        .def("ReadInput_HF", &ReadWriteFiles::ReadInput_HF)
        .def("ReadTokyo", &ReadWriteFiles::ReadTokyo)
        // .def("Read_KShell_HF_input", &ReadWriteFiles::Read_KShell_HF_input);
        .def("Read_KShell_HF_input",
             // Binding for the version without the Ref parameter
             [](ReadWriteFiles &self, const std::string &filename, ModelSpace &ms, Hamiltonian &inputH) {
                 self.Read_KShell_HF_input(filename, ms, inputH);
             }, py::arg("filename"), py::arg("ms"), py::arg("inputH"))
        .def("Read_KShell_HF_input",
             // Binding for the version with the Ref parameter
             [](ReadWriteFiles &self, const std::string &filename, ModelSpace &ms, Hamiltonian &inputH, const std::string &Ref) {
                 self.Read_KShell_HF_input(filename, ms, inputH, Ref);
             }, py::arg("filename"), py::arg("ms"), py::arg("inputH"), py::arg("Ref") = "")

        .def("Read_KShell_PHF_input",
             // Binding for the version with the Ref parameter
             [](ReadWriteFiles &self, const std::string &filename, ModelSpace &ms, Hamiltonian &inputH, const std::string &Ref) {
                 self.Read_KShell_PHF_input(filename, ms, inputH, Ref);
             }, py::arg("filename"), py::arg("ms"), py::arg("inputH"), py::arg("Ref") = "");


    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def(py::init<>())
        .def(py::init<ModelSpace &>(), py::keep_alive<1,2>())
        .def(py::init<const Hamiltonian&>(), py::keep_alive<1,2>()) // Copy constructor
        .def("SetMemoryLimitMB", [](Hamiltonian &h, double mb) {
            if (!std::isfinite(mb) || mb < 0) throw std::invalid_argument("memory limit must be finite and nonnegative");
            h.MSMEs.memory_limit_mb=mb;
        })
        .def("Prepare_MschemeH_Unrestricted", &Hamiltonian::Prepare_MschemeH_Unrestricted)
        .def("PrintHamiltonianInfo_pn", &Hamiltonian::PrintHamiltonianInfo_pn);


    py::class_<HartreeFock>(m, "HartreeFock")
        .def(py::init<Hamiltonian &>(), py::keep_alive<1,2>())
        .def("hybrid_state", &HFHybridAccess::state)
        .def("hybrid_evaluate", &HFHybridAccess::evaluate)
        .def("hybrid_response", &HFHybridAccess::response)
        .def("hybrid_operators", &HFHybridAccess::operators)
        .def("hybrid_basis", &HFHybridAccess::basis)
        .def("hybrid_multipole", &HFHybridAccess::multipole, py::arg("rank"), py::arg("mu"), py::arg("power"))
        .def("hybrid_accept", &HFHybridAccess::accept)
        .def("hybrid_reference", &HFHybridAccess::reference)
        .def("UpdateTolerance", &HartreeFock::UpdateTolerance)
        .def("Solve_diag", &HartreeFock::Solve_diag)
        .def("Solve_hybrid", &HartreeFock::Solve_hybrid)
        .def("Solve_broyden", &HartreeFock::Solve_broyden,
             py::arg("history_size") = 8, py::arg("alpha") = 0.5, py::arg("w0") = 0.01)
        .def("Solve_broyden_Constraint", &HartreeFock::Solve_broyden_Constraint,
             py::arg("history_size") = 8, py::arg("alpha") = 0.2, py::arg("w0") = 0.01,
             py::arg("constraint_strength") = 0.1)
        .def("Solve_gradient", &HartreeFock::Solve_gradient)
        .def("Solve_gradient_Constraint", &HartreeFock::Solve_gradient_Constraint)
        .def("Solve_hybrid_Constraint", &HartreeFock::Solve_hybrid_Constraint)
        .def("HF_ShapeCoefficients_Lab", &HartreeFock::HF_ShapeCoefficients_Lab)
        .def("Print_Jz", &HartreeFock::Print_Jz)
        .def("SaveHoleParameters", &HartreeFock::SaveHoleParameters)
        .def("RandomTransformationU", &HartreeFock::RandomTransformationU)
        .def("PrintFockMatrix", &HartreeFock::PrintFockMatrix)
        .def("PrintOccupationHO", &HartreeFock::PrintOccupationHO)
        .def("PrintOccupationHO_jorbit", &HartreeFock::PrintOccupationHO_jorbit)
        .def("PrintParameters_Hole", &HartreeFock::PrintParameters_Hole)
        .def("SetMaxIteration", &HartreeFock::SetMaxIteration)
        .def("UpdateGradientStepSize", &HartreeFock::UpdateGradientStepSize)
        .def("Reset_U", &HartreeFock::Reset_U);       


    py::class_<AngMomProjection>(m, "AngMomProjection")
        .def(py::init<>())
        .def(py::init<ModelSpace &>(), py::keep_alive<1,2>())
        .def("PrintInfo", &AngMomProjection::PrintInfo)
        .def("InitInt_HF_Projection", &AngMomProjection::InitInt_HF_Projection);



    py::class_<GCM_Projection>(m, "GCM_Projection")
        .def(py::init<>())
        .def(py::init<ModelSpace&, Hamiltonian&, AngMomProjection&>())
        .def("ReadBasis", &GCM_Projection::ReadBasis)
        .def("PrintResults", &GCM_Projection::PrintResults)
        .def("Do_Projection", &GCM_Projection::Do_Projection)
        .def("PrintInfo", &GCM_Projection::PrintInfo);


    // mpi functions
    m.def("mpi_initialize", &mpi_initialize, "Initialize MPI environment");
    m.def("mpi_finalize", &mpi_finalize, "Finalize MPI environment");


}
