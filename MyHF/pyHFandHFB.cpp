#include <Python.h>

#include "HartreeFock.h"
#include <string>
#include <sstream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>


namespace py = pybind11;

PYBIND11_MODULE(pyHFAndHFB, m)
{
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
        .def("PrintAllParameters_HF", &ModelSpace::PrintAllParameters_HF);

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
             }, py::arg("filename"), py::arg("ms"), py::arg("inputH"), py::arg("Ref") = "");


    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def(py::init<>())
        .def(py::init<ModelSpace &>())
        .def(py::init<const Hamiltonian&>()) // Copy constructor
        .def("Prepare_MschemeH_Unrestricted", &Hamiltonian::Prepare_MschemeH_Unrestricted)
        .def("PrintHamiltonianInfo_pn", &Hamiltonian::PrintHamiltonianInfo_pn);

    py::class_<HartreeFock>(m, "HartreeFock")
        .def(py::init<Hamiltonian &>())
        .def("Solve_diag", &HartreeFock::Solve_diag)
        .def("Solve_hybrid", &HartreeFock::Solve_hybrid)
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
        .def("Reset_U", &HartreeFock::Reset_U);       
}