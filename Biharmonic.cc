/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2005 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <math.h>

#include <deal.II/base/tensor_function.h>

namespace Biharmonic
{
  using namespace dealii;



  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem(const unsigned int degree, double young,
    		double thick, double poisson, double ext_Pressure);
    void run();

  private:
    void make_grid_and_dofs();
    void assemble_system();
    void solve();
    void output_results() const;

    const unsigned int degree;
    double rigidity, ext_Pressure, poisson;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double> solution;
    BlockVector<double> system_rhs;
  };



  namespace PrescribedSolution
  {

    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide(double ext_Pr)
        : Function<dim>(1)
      {ext_Pressure = ext_Pr;}

      virtual double value(const Point<dim> & p,
                           const unsigned int component = 0) const override;
      double ext_Pressure;
    };


    template <int dim>
    double RightHandSide<dim>::value(const Point<dim> & /*p*/,
                                     const unsigned int /*component*/) const
    {
      return ext_Pressure;
    }


  } // namespace PrescribedSolution




  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem(const unsigned int degree,
		  double young, double thick, double poisson, double ext_Pressure)
    : degree(degree)
    , fe(FE_Q<dim>(degree), 1, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
    , rigidity(young * pow(thick, 3) / (12 * (1-pow(poisson,2))))
    , ext_Pressure(ext_Pressure)
    , poisson(poisson)
  {}



  template <int dim>
  void MixedLaplaceProblem<dim>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);

    for (auto &face : triangulation.active_face_iterators())
      {
        if (std::fabs(face->center()(1) - (-1.0)) < 1e-12 ||
            std::fabs(face->center()(1) - (1.0)) < 1e-12  ||
        	std::fabs(face->center()(0) - (-1.0)) < 1e-12 ||
			std::fabs(face->center()(0) - (1.0)) < 1e-12)
        	face->set_boundary_id(0);
      }

    triangulation.refine_global(5);

    dof_handler.distribute_dofs(fe);

    DoFRenumbering::component_wise(dof_handler);

    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
    const unsigned int n_w = dofs_per_component[0],
                       n_u = dofs_per_component[1];

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_w << ')' << std::endl;

    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit(n_w, n_w);
    dsp.block(1, 0).reinit(n_u, n_w);
    dsp.block(0, 1).reinit(n_w, n_u);
    dsp.block(1, 1).reinit(n_u, n_u);
    dsp.collect_sizes();
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(2);
    solution.block(0).reinit(n_w);
    solution.block(1).reinit(n_u);
    solution.collect_sizes();

    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_w);
    system_rhs.block(1).reinit(n_u);
    system_rhs.collect_sizes();
  }



  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_system()
  {
    QGauss<dim>     quadrature_formula(degree + 2);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);


    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const PrescribedSolution::RightHandSide<dim> right_hand_side(ext_Pressure);

    std::vector<double>         rhs_values(n_q_points);

    const FEValuesExtractors::Scalar moments(0);
    const FEValuesExtractors::Scalar displacements(1);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
    	fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);


        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> grad_phi_i_u = fe_values[displacements].gradient(i, q);
              const Tensor<1, dim> grad_phi_i_w = fe_values[moments].gradient(i, q);

              const double phi_i_w     = fe_values[moments].value(i, q);
              const double phi_i_u     = fe_values[displacements].value(i, q);


              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> grad_phi_j_u = fe_values[displacements].gradient(j, q);
                  const Tensor<1, dim> grad_phi_j_w = fe_values[moments].gradient(j, q);
                  const double phi_j_u     = fe_values[displacements].value(j, q);
                  const double phi_j_w     = fe_values[moments].value(j, q);


                  local_matrix(i, j) +=
                    ( rigidity * grad_phi_i_w * grad_phi_j_u
                     - phi_j_w * phi_i_w//
                      + grad_phi_i_u * grad_phi_j_w)                //
                    * fe_values.JxW(q);
                }

              local_rhs(i) += phi_i_u * rhs_values[q] * fe_values.JxW(q);

            }

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              local_matrix(i, j));

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);

      }

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
											 Functions::ZeroFunction<dim>(dim),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);

  }



  template <int dim>
  void MixedLaplaceProblem<dim>::solve()
  {

//	           The UMFPACK Solver

      deallog << "Solving linear system... ";

	    SparseDirectUMFPACK A_direct;

	    A_direct.initialize(system_matrix);

	    A_direct.vmult(solution, system_rhs);

//
//    const auto &M = system_matrix.block(0, 0);
//    const auto &B = system_matrix.block(0, 1);
//    const auto &BC = system_matrix.block(1, 1);
//
//
//    const auto &F = system_rhs.block(1);
//
//    auto &W = solution.block(0);
//    auto &U = solution.block(1);
//
//    const auto op_M = linear_operator(M);
//    const auto op_B = linear_operator(B);
//    const auto op_BC = linear_operator(BC);
//
//
//    ReductionControl         reduction_control_M(2000, 1.0e-18, 1.0e-10);
//    SolverCG<Vector<double>> solver_M(reduction_control_M);
//    PreconditionJacobi<SparseMatrix<double>> preconditioner_M;
//
//    preconditioner_M.initialize(M);
//
//    const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);
//
//    const auto op_S = transpose_operator(op_B) * op_M_inv * op_B - op_BC;
//    const auto op_aS =
//      transpose_operator(op_B) * linear_operator(preconditioner_M) * op_B - op_BC;
//
//    IterationNumberControl   iteration_number_control_aS(30, 1.e-18);
//    SolverCG<Vector<double>> solver_aS(iteration_number_control_aS);
//
//    const auto preconditioner_S =
//      inverse_operator(op_aS, solver_aS, PreconditionIdentity());
//
//    const auto schur_rhs = F - 2 * F;
//
//    SolverControl            solver_control_S(2000, 1.e-12);
//    SolverCG<Vector<double>> solver_S(solver_control_S);
//
//    const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);
//
//    U = op_S_inv * schur_rhs;
//
//    std::cout << solver_control_S.last_step()
//              << " CG Schur complement iterations to obtain convergence."
//              << std::endl;
//
//    W = op_M_inv * (op_B * U);
//
//    W = W - 2 * W;
  }



  template <int dim>
  void MixedLaplaceProblem<dim>::output_results() const
  {
	std::vector<std::string> solution_names(1, "w");
    solution_names.emplace_back("u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(1,
                     DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation);

    data_out.build_patches(degree + 1);


    QGauss<dim>     quadrature_formula(degree + 2);

    Vector<double> dvdx2(triangulation.n_active_cells());
    Vector<double> dvdy2(triangulation.n_active_cells());

    Vector<double> moments(triangulation.n_active_cells());
    std::vector< std::vector< Tensor< 2, dim>>> moments_tensor(quadrature_formula.size(),
    		std::vector<Tensor<2, dim>>(dim));


    FEValues<dim>     fe_values(fe,
                                quadrature_formula,
                                update_values | update_hessians);


      for (auto &cell : dof_handler.active_cell_iterators())
          {

				double vx = 0., vy = 0.;
				fe_values.reinit(cell);
				fe_values.get_function_hessians(solution, moments_tensor);

				for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
					{
						vx += moments_tensor[q][1][0][0];
						vy += moments_tensor[q][1][1][1];
					}

				dvdx2(cell->active_cell_index()) =
				  (vx / quadrature_formula.size());
				dvdy2(cell->active_cell_index()) =
				  (vy / quadrature_formula.size());
          }


    moments = dvdx2 + poisson * dvdy2;

    data_out.add_data_vector(moments, "Bending_moment");
    data_out.build_patches();

    std::ofstream output("solution.vtu");
    data_out.write_vtu(output);

  }



  template <int dim>
  void MixedLaplaceProblem<dim>::run()
  {
    make_grid_and_dofs();
    assemble_system();
    solve();
    output_results();
  }
} // namespace Biharmonic



int main()
{
  try
    {
      using namespace Biharmonic;

      const unsigned int     fe_degree = 2;
      double young = 2.1e11, thick = 0.006, poisson = .33;
      MixedLaplaceProblem<2> mixed_laplace_problem(fe_degree, young, thick, poisson, 3000);
      mixed_laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
