namespace detail {
std::uint64_t N = 4096;  // cmd argument
size_t repetitions = 10; // cmd argument
size_t repetitions_pvn_compare = 10;
size_t restarts = 1;
bool use_pvn = true;
std::vector<double> A;
std::vector<double> B;
std::vector<double> C_reference;
double duration_kernel;
double gflops_kernel;
std::vector<size_t> thread_values;
size_t native_vector_width;
std::map<std::string, std::string> pvn_values_map;

std::function<bool(const std::vector<double> &C)> test_result =
    [&](const std::vector<double> &C) -> bool {
  for (size_t i = 0; i < N * N; i++) {
    double threshold = 1E-8;
    if (fabs(C[i] - C_reference[i]) >= threshold) {
      std::cout << "test error C: " << C[i] << " C_ref: " << C_reference[i]
                << " i: " << i << " (threshold: " << threshold << ")"
                << std::endl;
      return false;
    }
  }
  return true;
};
} // namespace detail

auto parameter_values_adjust_functor =
    [&](autotune::parameter_value_set &parameter_values) -> void {
  double X_REG = stod(parameter_values["X_REG"]);
  double Y_BASE_WIDTH = stod(parameter_values["Y_BASE_WIDTH"]);
  double L1_X = stod(parameter_values["L1_X"]);
  double L1_Y = stod(parameter_values["L1_Y"]);
  double L1_K = stod(parameter_values["L1_K"]);
  double L3_X = stod(parameter_values["L3_X"]);
  double L3_Y = stod(parameter_values["L3_Y"]);
  double L3_K = stod(parameter_values["L3_K"]);

  const double Y_REG =
      Y_BASE_WIDTH * static_cast<double>(detail::native_vector_width);

  // register parameters are always correct, never changed
  L1_X = autotune::detail::round_to_nearest_nonzero(L1_X, X_REG);
  L3_X = autotune::detail::round_to_nearest_nonzero(L3_X, L1_X);

  L1_Y = autotune::detail::round_to_nearest_nonzero(L1_Y, Y_REG);
  L3_Y = autotune::detail::round_to_nearest_nonzero(L3_Y, L1_Y);

  L3_K = autotune::detail::round_to_nearest_nonzero(L3_K, L1_K);

  parameter_values["L1_X"] = autotune::detail::truncate_trailing_zeros(L1_X);
  parameter_values["L3_X"] = autotune::detail::truncate_trailing_zeros(L3_X);

  parameter_values["L1_Y"] = autotune::detail::truncate_trailing_zeros(L1_Y);
  parameter_values["L3_Y"] = autotune::detail::truncate_trailing_zeros(L3_Y);

  parameter_values["L3_K"] = autotune::detail::truncate_trailing_zeros(L3_K);
};

template <typename parameter_set_type, typename F>
void evaluate_pvn(parameter_set_type &parameters, std::ofstream &pvn_csv_file,
                  const std::string &key, F &parameter_values_adjust_functor) {
  std::cout << "parameter values:" << std::endl;
  parameters.print_values();
  std::cout << "evaluate_pvn: " << key << std::endl;
  std::string old_value;
  if (key.compare("None") != 0) {
    old_value = parameters.get_by_name(key)->get_value();
    parameters.get_by_name(key)->set_value_unsafe(detail::pvn_values_map[key]);
  }
  autotune::parameter_value_set adjusted = to_parameter_values(parameters);
  parameter_values_adjust_functor(adjusted);
  autotune::combined_kernel.set_parameter_values(adjusted);
  std::cout << "adjusted parameter values:" << std::endl;
  autotune::print_parameter_values(adjusted);
  std::vector<double> C = autotune::combined_kernel(
      detail::N, detail::A, detail::B, detail::repetitions_pvn_compare,
      detail::duration_kernel, detail::gflops_kernel);
  bool test_ok = detail::test_result(C);
  if (test_ok) {
    std::cout << "optimal parameters test ok!" << std::endl;
  } else {
    std::cout << "optimal parameters FAILED test!" << std::endl;
  }
  for (size_t i = 0; i < parameters.size(); i += 1) {
    pvn_csv_file << adjusted[parameters[i]->get_name()] << ",";
  }
  pvn_csv_file << key << "," << detail::duration_kernel << ","
               << detail::gflops_kernel << std::endl;
  if (key.compare("None") != 0) {
    parameters.get_by_name(key)->set_value_unsafe(old_value);
  }
}

template <typename parameter_set_type, typename F>
void evaluate_pvn_group(parameter_set_type &parameters,
                        std::ofstream &pvn_csv_file,
                        const std::string &group_name,
                        const std::vector<std::string> &keys,
                        F &parameter_values_adjust_functor) {
  std::cout << "evaluate_pvn_group: ";
  for (const std::string &key : keys) {
    std::cout << key << " ";
  }
  std::cout << std::endl;
  std::vector<std::string> old_values;
  if (keys.size() > 0) {
    for (const std::string &key : keys) {
      old_values.push_back(parameters.get_by_name(key)->get_value());
    }
    for (const std::string &key : keys) {
      parameters.get_by_name(key)->set_value_unsafe(
          detail::pvn_values_map[key]);
    }
  }
  autotune::parameter_value_set adjusted = to_parameter_values(parameters);
  parameter_values_adjust_functor(adjusted);
  autotune::combined_kernel.set_parameter_values(adjusted);
  std::vector<double> C = autotune::combined_kernel(
      detail::N, detail::A, detail::B, detail::repetitions_pvn_compare,
      detail::duration_kernel, detail::gflops_kernel);
  bool test_ok = detail::test_result(C);
  if (test_ok) {
    std::cout << "optimal parameters test ok!" << std::endl;
  } else {
    std::cout << "optimal parameters FAILED test!" << std::endl;
  }
  for (size_t i = 0; i < parameters.size(); i += 1) {
    pvn_csv_file << adjusted[parameters[i]->get_name()] << ",";
  }
  pvn_csv_file << group_name << "," << detail::duration_kernel << ","
               << detail::gflops_kernel << std::endl;
  if (keys.size() > 0) {
    for (size_t i = 0; i < keys.size(); i += 1) {
      parameters.get_by_name(keys[i])->set_value_unsafe(old_values[i]);
    }
  }
}

template <typename parameter_set_type, typename F>
void pvn_compare(const std::string &scenario_name,
                 parameter_set_type &parameters,
                 F &parameter_values_adjust_functor) {
  // disable L3 by setting them to value of L2 each
  // disable L2 by setting them to value of L1 each
  // disable L1 by setting them to min
  // disable X_REG by setting it to 1
  // disable Y_BASE_WIDTH by setting it to 1
  std::cout << "starting pvn_compare..." << std::endl;

  std::ofstream pvn_csv_file(scenario_name + "_pvn_compare.csv");

  for (size_t i = 0; i < parameters.size(); i += 1) {
    pvn_csv_file << parameters[i]->get_name() << ",";
  }
  pvn_csv_file << "changed_key,duration,gflops" << std::endl;

  evaluate_pvn(parameters, pvn_csv_file, "None",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "KERNEL_NUMA",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "KERNEL_SCHEDULE",
               parameter_values_adjust_functor);
  // register parameters
  evaluate_pvn(parameters, pvn_csv_file, "X_REG",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "Y_BASE_WIDTH",
               parameter_values_adjust_functor);
  // L1 parameters
  evaluate_pvn(parameters, pvn_csv_file, "L1_X",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L1_Y",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L1_K",
               parameter_values_adjust_functor);
  evaluate_pvn_group(parameters, pvn_csv_file, "L1_GROUP",
                     {"L1_X", "L1_Y", "L1_K"}, parameter_values_adjust_functor);
  // L3 parameters
  evaluate_pvn(parameters, pvn_csv_file, "L3_X",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L3_Y",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L3_K",
               parameter_values_adjust_functor);

  evaluate_pvn_group(parameters, pvn_csv_file, "REG_GROUP",
                     {"X_REG", "Y_BASE_WIDTH"},
                     parameter_values_adjust_functor);

  evaluate_pvn_group(parameters, pvn_csv_file, "L3_GROUP",
                     {"L3_X", "L3_Y", "L3_K"}, parameter_values_adjust_functor);
  // other parameters
  evaluate_pvn(parameters, pvn_csv_file, "KERNEL_OMP_THREADS",
               parameter_values_adjust_functor);
}

template <typename parameter_set_type, typename tuner_t, typename F>
void do_tuning(tuner_t &tuner, parameter_set_type &ps,
               const std::string &scenario_name, const std::string &tuner_name,
               F &parameter_values_adjust_functor) {
  std::cout << "----------------- starting tuning, scenario name: "
            << scenario_name << " ------------ " << std::endl;
  for (size_t restart = 0; restart < detail::restarts; restart++) {
    std::cout << "restart: " << restart << std::endl;
    if (!detail::use_pvn) {
      bool valid_start_found = false;
      while (!valid_start_found) {
        for (size_t parameter_index = 0; parameter_index < ps.size();
             parameter_index++) {
          auto &p = ps[parameter_index];
          p->set_random_value();
        }
        valid_start_found = true; // should get adjusted
      }
    } else {
      for (auto &pair : detail::pvn_values_map) {
        ps.get_by_name(pair.first)->set_value_unsafe(pair.second);
      }
    }

    tuner.set_verbose(true);
    tuner.set_write_measurement(scenario_name + std::string("_") +
                                std::to_string(restart));
    tuner.setup_test(detail::test_result);

    parameter_set_type optimal_parameters;
    {
      std::chrono::high_resolution_clock::time_point start =
          std::chrono::high_resolution_clock::now();
      optimal_parameters =
          tuner.tune(detail::N, detail::A, detail::B, detail::repetitions,
                     detail::duration_kernel, detail::gflops_kernel);
      std::chrono::high_resolution_clock::time_point end =
          std::chrono::high_resolution_clock::now();
      double tuning_duration =
          std::chrono::duration<double>(end - start).count();
      tuner_duration_file << tuner_name << ", " << tuning_duration << std::endl;
    }
    autotune::parameter_value_set pv =
        autotune::to_parameter_values(optimal_parameters);
    autotune::parameter_values_to_file(pv, scenario_name + std::string("_") +
                                               std::to_string(restart) +
                                               std::string(".json"));

    std::cout << "----------------------- end tuning -----------------------"
              << std::endl;

    pvn_compare(scenario_name + std::string("_") + std::to_string(restart),
                optimal_parameters, parameter_values_adjust_functor);

    std::cout
        << "----------------------- end pvn compare -----------------------"
        << std::endl;
    std::cout << "optimal parameter values:" << std::endl;
    optimal_parameters.print_values();
    // autotune::combined_kernel.set_parameter_values(optimal_parameters);
    // autotune::combined_kernel.compile();

    // std::vector<double> C = autotune::combined_kernel(
    //     detail::N, detail::A, detail::B, detail::repetitions,
    //     detail::duration_kernel, detail::gflops_kernel);
    // bool test_ok = detail::test_result(C);
    // if (test_ok) {
    //   std::cout << "optimal parameters test ok!" << std::endl;
    // } else {
    //   std::cout << "optimal parameters FAILED test!" << std::endl;
    // }
  }
}
