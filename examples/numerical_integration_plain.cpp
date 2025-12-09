#include "hep/mc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::uint64_t samples = 1000000ULL;
    std::vector<std::size_t> threads{1, 2, 4};
    int repeats = 3;
    std::uint64_t seed = 123456789ULL;
};

double to_ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

std::vector<std::size_t> parse_thread_list(const std::string& arg) {
    std::vector<std::size_t> out;
    std::stringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(static_cast<std::size_t>(std::stoul(token)));
        }
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto require_value = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value after ") + name);
            }
            return std::string(argv[++i]);
        };

        if (a == "--samples") {
            opts.samples = std::stoull(require_value("--samples"));
        } else if (a == "--threads") {
            opts.threads = parse_thread_list(require_value("--threads"));
        } else if (a == "--repeats") {
            opts.repeats = std::stoi(require_value("--repeats"));
        } else if (a == "--seed") {
            opts.seed = std::stoull(require_value("--seed"));
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: ./numerical_integration_plain [--samples N] [--threads t1,t2] "
                         "[--repeats R] [--seed S]\n";
            std::exit(0);
        }
    }
    if (opts.threads.empty()) {
        opts.threads.push_back(1);
    }
    if (opts.samples == 0) {
        throw std::runtime_error("samples must be > 0");
    }
    return opts;
}

double square(hep::mc_point<double> const& x) {
    double v = x.point()[0];
    return v * v;
}

struct PartialSums {
    std::size_t calls = 0;
    std::size_t non_zero_calls = 0;
    std::size_t finite_calls = 0;
    double sum = 0.0;
    double sum_of_squares = 0.0;
};

hep::mc_result<double> combine_results(std::vector<PartialSums> const& partials) {
    std::size_t calls = 0;
    std::size_t non_zero = 0;
    std::size_t finite = 0;
    double sum = 0.0;
    double sumsq = 0.0;

    for (auto const& p : partials) {
        calls += p.calls;
        non_zero += p.non_zero_calls;
        finite += p.finite_calls;
        sum += p.sum;
        sumsq += p.sum_of_squares;
    }

    return hep::mc_result<double>(calls, non_zero, finite, sum, sumsq);
}

struct CsvRow {
    std::string section;
    std::size_t threads;
    int run_idx;
    std::uint64_t samples;
    double elapsed_ms;
    double throughput;
    double estimate;
    double variance;
};

CsvRow run_integration(std::size_t requested_threads, int run_idx, Options const& opts) {
    std::size_t active_threads = std::max<std::size_t>(
        1, std::min<std::size_t>(requested_threads, static_cast<std::size_t>(opts.samples)));

    std::vector<PartialSums> partials(active_threads);
    std::vector<std::thread> workers;
    workers.reserve(active_threads);

    std::size_t base_calls = static_cast<std::size_t>(opts.samples / active_threads);
    std::size_t remainder = static_cast<std::size_t>(opts.samples % active_threads);

    auto start = Clock::now();

    for (std::size_t idx = 0; idx < active_threads; ++idx) {
        std::size_t calls = base_calls + (idx < remainder ? 1 : 0);
        std::uint64_t seed = opts.seed + static_cast<std::uint64_t>(run_idx * 1315423911ULL + idx * 2654435761ULL);

        workers.emplace_back([idx, calls, seed, &partials]() {
            auto integrand = hep::make_integrand<double>(square, 1);
            std::mt19937_64 rng(seed);
            auto result = hep::plain_iteration(integrand, calls, rng);

            partials[idx].calls = result.calls();
            partials[idx].non_zero_calls = result.non_zero_calls();
            partials[idx].finite_calls = result.finite_calls();
            partials[idx].sum = result.sum();
            partials[idx].sum_of_squares = result.sum_of_squares();
        });
    }

    for (auto& t : workers) {
        t.join();
    }

    auto end = Clock::now();

    auto combined = combine_results(partials);

    double elapsed_ms = to_ms(end - start);
    double throughput = combined.calls() / (elapsed_ms / 1000.0);

    return CsvRow{
        "hep_plain_mt19937_64",
        active_threads,
        run_idx,
        opts.samples,
        elapsed_ms,
        throughput,
        combined.value(),
        combined.variance()
    };
}

void print_row(CsvRow const& row) {
    std::cout << row.section << ","
              << row.threads << ","
              << row.run_idx << ","
              << row.samples << ","
              << std::fixed << std::setprecision(4) << row.elapsed_ms << ","
              << std::fixed << std::setprecision(2) << row.throughput << ","
              << std::fixed << std::setprecision(6) << row.estimate << ","
              << std::fixed << std::setprecision(6) << row.variance
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);

        std::cout << "section,threads,run,samples,elapsed_ms,throughput,estimate,variance\n";

        for (std::size_t threads : opts.threads) {
            for (int run = 0; run < opts.repeats; ++run) {
                print_row(run_integration(threads, run, opts));
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "hep-mc numerical integration failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
