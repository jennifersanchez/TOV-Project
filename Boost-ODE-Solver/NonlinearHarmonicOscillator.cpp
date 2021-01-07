#include <array>                     // for std::array
#include <boost/numeric/odeint.hpp>  // for all boost ODE int
#include <functional>                // for std::ref
#include <iostream>                  // for std::cout
#include <vector>                    // for std::vector

static constexpr size_t number_of_dependent_variables = 2;

// The Observer is a class that stores the state of the ODE integration so we
// can get not just the end result of the integration, but a solution over time.
// That is, we have x(t) instead of just x(t_final).
class Observer {
 public:
  void operator()(
      const std::array<double, number_of_dependent_variables>& current_x,
      const double current_time) noexcept {
    x.push_back(current_x[0]);
    //solution for second equation, x[1]
    y.push_back(current_x[1]);
    time.push_back(current_time);
  }
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> time;
};

int main() {
  const double x_start_value = 0.0;
  const double x_end_value = 15.0;
  // The Delta x is somewhat arbitrary. This is basically the size of the
  // rectangles in a Riemann sum. So if you remember doing Riemann sums to
  // estimate an integral by cutting up the area under the curve into a bunch of
  // rectangles, initial_delta_x_for_integration is the width of those
  // rectangles.
  //
  // This also determines how often we store the current solution in the
  // observer object below.
  const double initial_delta_x_for_integration = 0.1;

  // Tolerances means how accurate we want the solution. Relative tolerance is
  // "how many digits" while absolute tolerence is "we only care about the
  // solution if its value is larger than this".
  //
  // Absolute tolerance (safeguards against zeros is the solution):
  // if y_numerical - y_exact < absolute_tolerance:
  //   "things are fine, we don't care about tiny numbers"
  const double absolute_tolerance = 1.0e-8;
  // Relative tolerance:
  // (y_numerical - y_exact) / abs(y_numerical)
  const double relative_tolerance = 1.0e-8;

  // All these super long types are just Boost being.... "flexible" (I'd say
  // difficult, actually)
  using StateDopri5 = boost::numeric::odeint::runge_kutta_dopri5<
      std::array<double, number_of_dependent_variables>>;
  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<StateDopri5>>
      dopri5 = make_dense_output(absolute_tolerance, relative_tolerance,
                                 StateDopri5{});

  // The observer object will store the result at specific times. Which times
  // can be controlled by choosing changing initial_delta_x_for_integration.
  Observer observer_fixed_step_size{};

  // The observer object will store the result at specific times. Which times
  // are specified in the times_to_observe_at std::vector<double>. The
  // integration range is from the first value to the last.
  Observer observer_at_chosen_steps{};
  std::vector<double> times_to_observe_at{
    x_start_value, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, x_end_value};

  // This is the initial condition and will be updated as we integrate.
  // x{{}}. we need to initial values. one for x and one for y. 
  //Outer brace is the initialization (in modern c++ it's called 'brace initialization')
  //while inner brace is list of things you want to initialize 
  std::array<double, number_of_dependent_variables> x{{1.0, 0.0}};
  // We want to solve:
  // dx / dt = x

  // Integrate while observing at constant step size
  boost::numeric::odeint::integrate_const(
      dopri5,
      [](const std::array<double, number_of_dependent_variables>&
             current_value_of_x,
         std::array<double, number_of_dependent_variables>&
             current_time_derivative_of_x,
         const double current_time_t) noexcept {
        // Note we don't use the time explicitly!
        (void)current_time_t;
        // This computes the dx/dt
	// x[0]is dx/dt = y. [1] = y while [0]=x. 
	// dy/dt = blah blah. 
        current_time_derivative_of_x[0] = current_value_of_x[1];
	current_time_derivative_of_x[1]= -cos(current_value_of_x[0]);
      },
      x, x_start_value, x_end_value, initial_delta_x_for_integration,
      std::ref(observer_fixed_step_size));

  std::cout << "Printing out solution obtained from the fixed step size "
               "observer.\n";
  for (size_t time_index = 0; time_index < observer_fixed_step_size.x.size();
       ++time_index) {
    std::cout //<< observer_fixed_step_size.time[time_index] << ", ";
    << observer_fixed_step_size.x[time_index] << ",";
  }

  // Need to reset to initial condition before integrating again
  x[0] = 1.0;
  x[1]= 0.0;

  // Integrate while observing at specified times
  boost::numeric::odeint::integrate_times(
      dopri5,
      [](const std::array<double, number_of_dependent_variables>&
             current_value_of_x,
         std::array<double, number_of_dependent_variables>&
             current_time_derivative_of_x,
         const double current_time_t) noexcept {
        // Note we don't use the time explicitly!
        (void)current_time_t;
        // This computes the dx/dt
        current_time_derivative_of_x[0] = current_value_of_x[1];
	current_time_derivative_of_x[1]= -cos(current_value_of_x[0]);
      },
      x, times_to_observe_at.begin(), times_to_observe_at.end(),
      initial_delta_x_for_integration, std::ref(observer_at_chosen_steps));

  std::cout
      << "\n\nPrinting out solution obtained at explicitly chosen times.\n";
  for (size_t time_index = 0; time_index < observer_at_chosen_steps.x.size();
       ++time_index) {
    std::cout << observer_at_chosen_steps.time[time_index] << " "
              << observer_at_chosen_steps.x[time_index] << "\n";
  }
}
