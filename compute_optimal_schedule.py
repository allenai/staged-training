# pythonw -m IPython

import math
import pprint

import matplotlib.pylab as plt
import numpy as np
import torch

# N = number non embedding params
# L = loss
# C = compute

# Table 5 in Appendix A
B_star = 2.1e8
S_c = 2.1e3
# S_c = 2.1e3 * 2.0  # this is from Eqn 1.6, which gives Smin(S) = S / (1 + B_crit / B), so Smin(S) = S / 2 at B==B_crit
N_c = 8.8e13
alpha_B = 0.21
alpha_N = 0.076
alpha_S = 0.76


def flops_to_pf_days(flops):
    # one PF-day = 10^15 × 24 × 3600 = 8.64 × 10^19 flops
    return flops / 8.64e19


def get_jacobian(x, func):
    # get the jacobian of the function at x
    # x == a numpy array
    # returns jacobian = [dfunc / dx[0], dfunc / dx[1], ...]
    x_tensor = torch.tensor(x, requires_grad=True)
    y = func(x_tensor)
    y.backward()
    return x_tensor.grad.numpy()


def numpy_wrap_torch(x, func):
    # call the torch func with numpy x and return numpy output
    return func(torch.tensor(x)).numpy()


# Equation 1.6:
# L(N, S) = (N_c / N) ^ alpha_N + (S_c / Smin(S)) ^ alpha_S
#   Smin(S) = S / (1 + Bcrit / B)
# If B == Bcrit, then Smin(S) = S / 2 -->
#   L(N, S) = (N_c / N) ^ alpha_N + (2 * S_c / S) ^ alpha_S


class MinimizeComputeFixedSize:
    # minimize Compute(number parameters, number steps)
    # for a given loss target L_target
    #
    # min_{N, S} compute = 6 * N * B_star / L^(1 / alpha_B) * S
    #   such that L(N, S) = L_target = (N_c / N) ^ alpha_N + (2 * S_c / S) ^ alpha_S
    #         1e6 <= N <= np.inf
    #         1 <= S <= np.inf
    #
    # for optimization need to define the function and jacobian

    # x = (np.log10(N), np.log10(S))

    def __init__(self, L_target):
        self.L_target = L_target

    def _get_N_S(self, x):
        return (10.0 ** x[0], 10.0 ** x[1])

    def compute(self, x):
        N, S = self._get_N_S(x)
        return flops_to_pf_days(6.0 * N * B_star / self.L_target ** (1.0 / alpha_B) * S)

    def jac_compute(self, x):
        return get_jacobian(x, self.compute)

    def constraint_func(self, x):
        # the constraint is an equality constraint constraint(x) = 0 = L(N, S) - L_target
        N, S = self._get_N_S(x)
        return (N_c / N) ** alpha_N + (2 * S_c / S) ** alpha_S - self.L_target

    def get_initial_conditions(self, N):
        # get the initial conditions that satisfy the constraint
        # L_target = (N_c / N) ** alpha_N + (S_c / S) ** alpha_S   ---->
        #      L_target - (N_c / N) ** alpha_N = (S_c / S) ** alpha_S
        #      (L_target - (N_c / N) ** alpha_N) ** (1.0 / alpha_S) = (S_c / S)
        #      S = S_c / ((L_target - (N_c / N) ** alpha_N) ** (1.0 / alpha_S))
        return 2 * S_c / ((self.L_target - (N_c / N) ** alpha_N) ** (1.0 / alpha_S))

    def constraint_jac(self, x):
        return get_jacobian(x, self.constraint_func)

    def solve(self):
        from scipy.optimize import Bounds, minimize

        N0 = 11  # 100B = a large number that can achieve any loss
        x0 = np.array([N0, np.log10(self.get_initial_conditions(10.0 ** N0))])

        constraint = {
            "type": "eq",
            "fun": self.constraint_func,
            "jac": self.constraint_jac,
        }
        # bounds takes two lists: all lower bounds and all upper bounds
        bounds = Bounds([6, 2], [np.inf, np.inf])

        result = minimize(
            self.compute,
            x0,
            method="SLSQP",
            jac=self.jac_compute,
            constraints=[constraint],
            options={"ftol": 1e-9, "disp": True},
            bounds=bounds,
        )

        print(result)

        print("Optimal N, S = 10**{}, 10**{}".format(result.x[0], result.x[1]))
        print("Minimum compute = {} PF-days".format(result.fun))


def critical_batch_size(L):
    # compute B_crit(L) = B_* / L ^ (1 / alpha_B)
    return B_star / L ** (1.0 / alpha_B)


class MinimizeComputeSchedule:
    # have a training schedule with sequence of model sizes and number of steps
    #   schedule = {[N_i, S_i], i = 1, ..., number of steps
    #   Can then compute total compute = SUM_i 6 * N_i * B_star / L^(1 / alpha_B) * S_i
    #
    # To compute the total loss at the end of stage k of training, do the following:
    #   loss_0 = loss at end of first stage, use formula from scaling laws:
    #            (N_c / N) ** alpha_N + (S_c / S) ** alpha_S
    #   loss_k = loss at end of stage k: assume the learning curve follows the same trajectory with step
    #           after increasing the model size
    #       (1) first compute effective number of steps that model size N_k would have taken to reach loss_{k-1}, the loss at end of previous stage, S_eff_{k-1}
    #       (2) Compute loss at S_k updates past S_eff_{k-1} = (N_c / N) ** alpha_N = (S_c / (S_eff_{k-1} + S_k) ** alpha_S
    #
    # then have the constraints that:
    #   0 <= S_i <= inf
    #   N_{i-1} <= N_i
    #   loss at end of training schedule = L_target
    #   N at the end of training = N_target  # optional
    #   N_increase_lb <= N_i / N_{i-1} <= N_increase_ub  # optional
    #
    # can recover the one step schedule by taking a single step on first stage, then setting all step sizes to 0 and model size the same as previous step
    #
    # we encode the schedule vector to optimize as:
    #   schedule[0] = log10(number parameters)
    #   schedule[1] = log10(number steps)

    #   schedule[2], 4, 6, .. (all even numbers): the INCREASE in number parameters from previous step, such that
    #       total parameters at step k = 10 ** (SUM(schedule[2 * (k-1)], k=1, ... schedule length)
    #   and total steps at step k = 10 ** (SUM(schedule[2 * (k-1) + 1], k=1, ... schedule length)
    # schedule = [log10(N_0), log10(S_0), DELTA N_1, DELTA S_1, ....]

    # To allow batch size to vary across the training run:
    #   If we assume we always train at the critical batch size, then the
    #   Loss calculation is unchanged (and the batch size dependence in the loss is quite small anyway,
    #   as it will vary between effectively scaling S_c by 1 --> 2^alpha_S = 1.7, but at the
    #   number of steps we are taking this only impacts a very small portion of steps
    #   near the beginning of training.
    #   But we can take the batch size into account in the Compute calculation by substituting
    #       B_star / L_k^(1/alpha_B) with each stages loss.

    def __init__(
        self,
        L_target,
        schedule_length=10,
        allow_batch_size_to_vary=False,
        N_target=None,
        N_increase_lb=None,
        N_increase_ub=None,
    ):
        self.L_target = L_target
        self.schedule_length = schedule_length
        self.allow_batch_size_to_vary = allow_batch_size_to_vary
        self.N_target = N_target
        self.N_increase_lb = N_increase_lb
        self.N_increase_ub = N_increase_ub

    def _torch_compute(self, schedule):
        # get the total compute used for the schedule
        # schedule = [N_0, S_0, N_1, S_1, ...]
        compute_per_stage, batch_size_per_stage = self._compute_stages(schedule)
        return sum(compute_per_stage)

    def _loss(self, N, S):
        # L(N, S) - assumes training at critical batch size
        return (N_c / N) ** alpha_N + (2 * S_c / S) ** alpha_S

    def _compute_stages(self, schedule):
        sizes_steps_losses = self.get_loss_at_each_stage(schedule)
        model_sizes = sizes_steps_losses["model_sizes"]
        total_steps = sizes_steps_losses["total_steps"]
        loss_stages = sizes_steps_losses["loss_stages"]

        compute_stages = []
        batch_size_stages = []
        for k in range(self.schedule_length):
            N = model_sizes[k]
            if k == 0:
                S = total_steps[0]
            else:
                # total steps for this stage of the schedule
                S = total_steps[k] - total_steps[k - 1]

            if self.allow_batch_size_to_vary:
                batch_size = B_star / loss_stages[k] ** (1.0 / alpha_B)
            else:
                batch_size = B_star / self.L_target ** (1.0 / alpha_B)
            batch_size_stages.append(batch_size)
            compute_stages.append(flops_to_pf_days(6.0 * N * batch_size * S))
        return compute_stages, batch_size_stages

    def compute(self, schedule):
        return numpy_wrap_torch(schedule, self._torch_compute)

    def jac_compute(self, schedule):
        return get_jacobian(schedule, self._torch_compute)

    def get_effective_number_steps_for_Ltarget_and_N(self, N, L_target):
        # get the effective number of steps S such that the loss is L_target
        # for model size N
        # WARNING: uses the passed value of L_target and not the self.value as in other functions!
        #
        # L_target = (N_c / N) ** alpha_N + (S_c / S) ** alpha_S   ---->
        #      L_target - (N_c / N) ** alpha_N = (S_c / S) ** alpha_S
        #      (L_target - (N_c / N) ** alpha_N) ** (1.0 / alpha_S) = (S_c / S)
        #      S = S_c / ((L_target - (N_c / N) ** alpha_N) ** (1.0 / alpha_S))
        return 2 * S_c / ((L_target - (N_c / N) ** alpha_N) ** (1.0 / alpha_S))

    def _unpack_schedule_to_model_sizes_and_total_steps(self, schedule):
        # Input schedule is the raw schedule
        # unpacks the even indices as the model sizes, computes total size, and 10 ** it
        # gets the odd indices as the steps for each stage and does the same
        raw_sizes = schedule.index_select(
            0, torch.tensor(range(0, 2 * self.schedule_length - 1, 2))
        )
        model_sizes = 10 ** torch.cumsum(raw_sizes, dim=0)

        raw_steps = schedule.index_select(
            0, torch.tensor(range(1, 2 * self.schedule_length, 2))
        )
        total_steps = 10 ** torch.cumsum(raw_steps, dim=0)

        return model_sizes, total_steps

    def get_loss_at_each_stage(self, schedule):
        # returns a vector of losses at the end of each stage in the schedule
        loss_stages = []

        # model size at the end of each schedule piece (in number of non-embedding params)
        # total number of steps at the end of each schedule piece
        model_sizes, total_steps = self._unpack_schedule_to_model_sizes_and_total_steps(
            schedule
        )

        # get the first loss
        N = model_sizes[0]
        S = total_steps[0]
        loss_0 = self._loss(model_sizes[0], total_steps[0])
        loss_stages.append(loss_0)

        last_loss = loss_0
        for k in range(1, self.schedule_length):
            # (1) get the effective number of steps at N_k to give last_loss = L_{k-1}
            S_eff_km1 = self.get_effective_number_steps_for_Ltarget_and_N(
                model_sizes[k], last_loss
            )
            loss_k = self._loss(
                model_sizes[k], S_eff_km1 + total_steps[k] - total_steps[k - 1]
            )
            loss_stages.append(loss_k)
            last_loss = loss_k

        return {
            "loss_stages": loss_stages,
            "model_sizes": model_sizes,
            "total_steps": total_steps,
        }

    def get_initial_conditions(self):
        # get the initial conditions that satisfy the constraints.
        # Use a single stage then empty remainder of the schedule
        N0 = 12  # 1 trillion parameters
        S0 = np.log10(
            self.get_effective_number_steps_for_Ltarget_and_N(10.0 ** N0, self.L_target)
        )

        schedule = np.zeros(self.schedule_length * 2)
        schedule[0] = N0
        schedule[1] = S0

        # all remaining steps have parameters = N0 and steps = 0
        return schedule

    def print_info_schedule(self, schedule, actual_batch_size):
        """
        actual_batch_size: print the schedule using the critical batch size and also an actual batch size
        """
        sched = torch.tensor(schedule)
        print("Raw schedule: ", sched)

        compute_per_stage, batch_size_per_stage = self._compute_stages(sched)
        total_compute = sum(compute_per_stage).item()
        results = self.get_loss_at_each_stage(sched)
        model_sizes = results["model_sizes"]
        total_steps = results["total_steps"]

        percent_increases = np.zeros(model_sizes.shape) * np.nan
        percent_increases[1:] = (model_sizes[1:] - model_sizes[0:-1]) / model_sizes[
            0:-1
        ]
        total_steps_per_stage = np.zeros(total_steps.shape) * np.nan
        total_steps_per_stage[0] = total_steps[0]
        total_steps_per_stage[1:] = total_steps[1:] - total_steps[:-1]
        print(
            "Loss, Model sizes, percent increase over previous stage, # steps_crit, compute, compute %, bsz_crit, # steps_actual"
        )
        for l, m, p, s, c, bsz in zip(
            results["loss_stages"],
            model_sizes,
            percent_increases,
            total_steps_per_stage,
            compute_per_stage,
            batch_size_per_stage,
        ):
            steps_actual = (
                s * bsz / actual_batch_size if actual_batch_size is not None else 0
            )
            print(
                f"{round(l.item(), 4)}, {int(m.item() / 1000000.0)}M, {round(p, 2)}, {int(s.item())}, {round(c.item(), 4)}PF-day, {round(c.item()/total_compute, 2)}, {int(bsz)}, {int(steps_actual / 1000)}K"
            )

    def _torch_constraint_func(self, x):
        loss_stages = self.get_loss_at_each_stage(x)["loss_stages"]
        return loss_stages[-1] - self.L_target

    def constraint_func(self, x):
        # the constraint that the loss at end of training is L_target
        # takes numpy as input and output
        return numpy_wrap_torch(x, self._torch_constraint_func)

    def constraint_jac(self, x):
        # numpy input / output
        return get_jacobian(x, self._torch_constraint_func)

    def _torch_constraint_N_target_func(self, x):
        model_sizes, _ = self._unpack_schedule_to_model_sizes_and_total_steps(x)
        final_model_size = model_sizes[-1]
        return final_model_size - self.N_target

    def constraint_N_target_func(self, x):
        # the constraint that the model size at end of training is N_target
        # takes numpy as input and output
        return numpy_wrap_torch(x, self._torch_constraint_N_target_func)

    def constraint_N_target_jac(self, x):
        # numpy input / output
        return get_jacobian(x, self._torch_constraint_N_target_func)

    def solve(self, actual_batch_size=None):
        from scipy.optimize import Bounds, minimize

        x0 = self.get_initial_conditions()

        # get the constraint that the loss at end of training is L_target
        constraint_L_target = {
            "type": "eq",
            "fun": self.constraint_func,
            "jac": self.constraint_jac,
        }
        constraints = [constraint_L_target]
        if self.N_target is not None:
            constraint_N_target = {
                "type": "eq",
                "fun": self.constraint_N_target_func,
                "jac": self.constraint_N_target_jac,
            }
            constraints.append(constraint_N_target)

        # due to the formulation of schedule in terms of deltas, the constraint N_i >= N_{i-1} is automatically satisfied

        # bounds takes two lists: all lower bounds and all upper bounds
        # all of the bounds are >= 0 and <= inf
        bounds = Bounds([0.0] * len(x0), [np.inf] * len(x0))
        for i in range(2, len(bounds.ub)):
            if i % 2 == 0:
                if self.N_increase_lb is not None:
                    bounds.lb[i] = math.log10(self.N_increase_lb)
                if self.N_increase_ub is not None:
                    bounds.ub[i] = math.log10(self.N_increase_ub)
        print(f"Bounds: {bounds}")

        result = minimize(
            self.compute,
            x0,
            method="SLSQP",
            jac=self.jac_compute,
            constraints=constraints,
            options={"ftol": 1e-9, "disp": True, "maxiter": 500},
            bounds=bounds,
        )

        print(result)
        print("Minimum compute = {} PF-days".format(result.fun))
        self.print_info_schedule(result.x, actual_batch_size=actual_batch_size)

        return result


def compare_schedule_to_fixed_size(
    L_target,
    max_schedule_length,
    allow_batch_size_to_vary=False,
    N_target=None,
    N_increase_lb=None,
    N_increase_ub=None,
    actual_batch_size=None,
):
    # Compare the optimal schedule to the optimal model size / number of steps for a single stage
    # Can accomplish this by computing schedule with stages = 1, 2, .., and comparing all to stage 1

    results = []
    for schedule_length in [1, max_schedule_length]:
        minimizer = MinimizeComputeSchedule(
            L_target,
            schedule_length,
            allow_batch_size_to_vary=allow_batch_size_to_vary,
            N_target=N_target,
            N_increase_lb=N_increase_lb,
            N_increase_ub=N_increase_ub,
        )
        result = minimizer.solve(actual_batch_size=actual_batch_size)
        results.append(result)
        print("========================")

    compute_improvements = [result.fun / results[0].fun for result in results]
    pprint.pprint(list(zip(range(1, max_schedule_length + 1), compute_improvements)))


# ## an analysis / plotting of amount of compute vs parameters and target loss,
# ignoring the learning dynamics


def compute_N_L(N, L):
    # eqn B.15 computes flops
    # returns compute in PF-days
    # returns NaN if it's not possible to achieve the given L with the number of parameters N
    flops_b_15 = (
        6
        * B_star
        * S_c
        * N
        / L ** (1 / alpha_B)
        * (L - (N_c / N) ** alpha_N) ** (-1.0 / alpha_S)
    )
    return flops_to_pf_days(flops_b_15)


def plot_compute_N_for_fixed_L(L_target):
    # plot the amount of compute vs N for the given L_target value

    # one million to 100B, 1e6 --> 100e9 = 1e11
    N_range = 10.0 ** np.linspace(6, 11, 1000)
    compute_for_L_target = compute_N_L(N_range, L_target)

    plt.plot(np.log10(N_range), np.log10(compute_for_L_target))
    plt.title(
        "Amount of compute needed to reach L={}\nvs number parameters".format(L_target)
    )
    plt.xlabel("Number non-embedding parameters (log10 scale)")
    plt.ylabel("Number of PF-days compute, (log10 scale)")
    plt.show()


def plot_compute_vs_Ltarget_N():
    # plot the amount of compute vs N for all L_target values

    # one million to 100B, 1e6 --> 100e9 = 1e11
    N_range = 10.0 ** np.linspace(6, 11, 1000)
    L_target_range = np.linspace(6, 2, 9)

    compute = np.zeros((len(N_range), len(L_target_range)))
    for k, L_target in enumerate(L_target_range):
        compute[:, k] = compute_N_L(N_range, L_target)

    fig = plt.figure(1)
    for k, L_target in enumerate(L_target_range):
        plt.plot(
            np.log10(N_range), np.log10(compute[:, k]), label="{}".format(L_target)
        )
    plt.legend()

    plt.title("Amount of compute vs number of parameters for a given L=L_target")
    plt.xlabel("Number non-embedding parameters (log10 scale)")
    plt.ylabel("Number of PF-days compute, (log10 scale)")
    fig.show()
