#include "HartreeFock.h"

// The constrained modified-Broyden solver is implemented in
// HartreeFock_Broyden.cpp through HartreeFock::Solve_broyden_impl(...).
//
// This translation unit is intentionally kept as a stub because the makefile
// still builds HartreeFock_Broyden_Constraint.o.  Keeping an old independent
// definition of HartreeFock::Solve_broyden_Constraint(...) here causes a linker
// multiple-definition error.
