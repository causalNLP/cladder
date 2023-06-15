import numpy as np

from ..base import register_graph
from ..bayes import node, hparam
from .phenomena import Confounding
# from .phenomena import Confounding
np.random.seed(0)



@register_graph('simpson')
class SimpsonsParadox(Confounding):

    def aggregate_stats(self): # p(Y | X)
        '''p(Y | X) (calculated only from Y parameters, severity, and gap)'''

        agg_treated = self.marginals(X=1)['Y']
        agg_control = self.marginals(X=0)['Y']

        return agg_control, agg_treated



# @register_graph('simpson')
# class SimpsonsParadox(Phenomenon):
#     '''
#     Collider example
#
#     Z -> X
#     X, Z -> Y
#
#     X is treatment (=1) vs control (=0)
#     Z is group A (=1) vs group B (=0)
#
#     Groups A and B are referred to as the "best" and "worst" groups, since wlog,
#     group A is chosen to perform better than group B.
#     '''
#     description_template = hparam('We know that both {Xname} and {Zname} affect {Yname} but not each other.')
#
#     details_template = hparam('Furthermore, {Y1} is more likely than {Y0} when {X1} for both {Z1} and {Z0}. '
#                               'However, {Y1} is overall less likely than {Y0} when {X1} compared to {X0}.')
#
#
#     treatment_fails = hparam(False)  # if true, the treatment effect is reversed (i.e. treated group has lower outcomes)
#
#
#     @hparam
#     def overlap(self):  # between (0, 1) overlap between treated outcomes and control outcomes
#         # return self._rng.uniform(.3, .8)
#         return self._rng.choice([round(rd,1) for rd in np.linspace(0.3, 0.8, 6)])
#
#
#     @hparam
#     def width(self):  # between (0, 1) overall range of outcomes
#         # return self._rng.uniform(.2, .8)
#         return self._rng.choice([round(rd,1) for rd in np.linspace(0.2, 0.8, 7)])
#
#
#     @hparam
#     def offset(self):
#         # return self._rng.uniform(.1, .9)
#         return self._rng.choice([round(rd,1) for rd in np.linspace(0.1, 0.9, 9)])
#
#
#     @hparam
#     def overlap_offset(self):
#         # return self._rng.uniform(.3, .7)
#         return self._rng.choice([round(rd,1) for rd in np.linspace(0.3, 0.7, 5)])
#
#
#
#     @hparam
#     def prior(self):
#         '''probability of being in group A (best)'''
#         # return self._rng.uniform(.3, .7)
#         return self._rng.choice([round(rd,1) for rd in np.linspace(0.3, 0.7, 5)])
#     @hparam
#     def aggregate_offset(self):
#         '''aggregate_offset: between [0, 1] where in the overlap the aggregate outcomes are'''
#         # return self._rng.uniform(.3, .7)
#         return self._rng.choice([round(rd,1) for rd in np.linspace(0.3, 0.7, 5)])
#
#     @hparam
#     def gap(self):
#         '''
#         between [-1, 1] gap between the aggregate treatment and control outcome
#         (positive implies agg_control > agg_treated)
#         '''
#         # return self._rng.uniform(.2, .7)
#
#         return self._rng.choice([round(rd,1) for rd in np.linspace(0.2, 0.7, 6)])
#
#
#     # Z = node()
#     V1 = node()
#     X = node(V1)
#     Y = node(V1, X)
#
#
#     def __init__(self, *args, **kwargs):
#         print(*args, **kwargs)
#         super().__init__(*args, **kwargs)
#         self._set_simpsons_paradox_params()
#
#
#     def _set_simpsons_paradox_params(self):
#         offset = self.offset * (1 - self.width)
#         overlap_offset = self.overlap_offset * (1 - self.overlap) * self.width
#
#         worst_control = offset
#         worst_treated = offset + overlap_offset
#         best_control = offset + overlap_offset + self.overlap * self.width
#         best_treated = offset + self.width
#
#         params = [[worst_control, best_control], [worst_treated, best_treated]]
#
#         prior = self.prior
#
#         gap = self.gap
#         aggregate_offset = self.aggregate_offset
#
#         Y00, Y01 = params[0]
#         Y10, Y11 = params[1]
#
#         self.V1.param = prior
#
#         X0, X1 = -1, -1
#
#         fuel = 100
#         while X0 < 0 or X0 > 1 or X1 < 0 or X1 > 1:
#             agg_control, agg_treated = self._expected_aggregate(params, gap, aggregate_offset)
#             X0, X1 = self._solve_for_simpson_posterior(agg_control, agg_treated, Y00, Y01, Y10, Y11, prior)
#             fuel -= 1
#             if fuel == 0:
#                 self._set_simpsons_paradox_params()
#                 raise ValueError('Could not find a valid solution')
#             # gap *= 0.9
#             aggregate_offset += (0.5 - aggregate_offset) * 0.3
#             # aggregate_offset *= 0.1
#
#         self.X.param = [X0, X1]
#         self.Y.param = params
#
#         if self.treatment_fails:
#             self.Y.param = self.Y.param[:, ::-1]
#             self.X.param = self.X.param[::-1]
#
#
#     @staticmethod
#     def _solve_for_simpson_posterior(aggX0, aggX1, Y00, Y01, Y10, Y11, Z):
#         '''
#
#         :param aggX0: p(Y = 1 | X = 0)
#         :param aggX1: p(Y = 1 | X = 1)
#         :param Y00: p(Y = 1 | X = 0, Z = 0)
#         :param Y01: p(Y = 1 | X = 0, Z = 1)
#         :param Y10: p(Y = 1 | X = 1, Z = 0)
#         :param Y11: p(Y = 1 | X = 1, Z = 1)
#         :param Z: p(Z = 1)
#
#         :return: p(X = 1 | Z = 0), p(X = 1 | Z = 1)
#         '''
#
#         nZ = 1 - Z
#
#         a1, b1, c1 = (Y10 - aggX1) * nZ, (Y11 - aggX1) * Z, 0
#         a2, b2, c2 = (aggX0 - Y00) * nZ, (aggX0 - Y01) * Z, aggX0 - Y01 * Z - Y00 * nZ
#
#         det = a1 * b2 - a2 * b1
#         assert abs(det) > 1e-6, 'Determinant is too small'
#
#         xx = (b2 * c1 - b1 * c2) / det
#         yy = (a1 * c2 - a2 * c1) / det
#         return xx, yy
#
#
#     @staticmethod
#     def _expected_aggregate(params, gap, offset):
#         [[worst_control, best_control], [worst_treated, best_treated]] = params
#
#         mn = worst_treated
#         mx = best_control
#
#         overlap = mx - mn
#         gap = abs(gap) * overlap
#         offset_range = overlap - gap
#         offset = offset * offset_range
#
#         agg_treated = mn + offset
#         agg_control = agg_treated + gap
#
#         if gap < 0:
#             agg_control, agg_treated = agg_treated, agg_control
#
#         return agg_control, agg_treated
#
#
#     def _aggregate_stats(self): # p(Y | X)
#         '''p(Y | X) (calculated only from Y parameters, severity, and gap)'''
#
#         agg_treated = self.marginals(X=1)['Y']
#         agg_control = self.marginals(X=0)['Y']
#
#         return agg_control, agg_treated
#
#         # ((worst_control, best_control), (worst_treated, best_treated)) = self.Y.p
#         # # if self.treatment_fails:
#         # #     ((worst_treated, best_treated), (worst_control, best_control)) = self.Y.p
#         #
#         # mid_treated = best_control
#         # mid_control = worst_treated
#         #
#         # assert severity >= 0
#         # agg_treated = worst_treated * severity + mid_treated * (1 - severity)
#         #
#         # if gap > 0:
#         #     agg_control = best_control * gap + agg_treated * (1 - gap)
#         # else:
#         #     gap = -gap
#         #     agg_control = mid_control * gap + agg_treated * (1 - gap)
#         #
#         # # if self.treatment_fails:
#         # #     agg_treated, agg_control = agg_control, agg_treated
#         #
#         # return agg_control, agg_treated
#
#     def simpsons_paradox_stats(self):
#         ((worst_control, best_control), (worst_treated, best_treated)) = self.Y.param
#         agg_control, agg_treated = self._aggregate_stats()
#
#         stats = [[worst_control, best_control, agg_control],
#                  [worst_treated, best_treated, agg_treated]]
#
#         return np.array(stats)



# agg_control, agg_treated = self._aggregate_stats()
#
# alpha = (best_treated - agg_treated) / (best_treated - worst_treated)
# beta = (agg_control - worst_control) / (best_control - worst_control)
#
# assert self.prior == 0.5  # TODO: generalize to priors other than 0.5
#
# self.Z.p = self.prior
# self.X.p = [self.severity + min(self.severity, 1 - self.severity) * self.gap, self.severity]
# # self.X.p = [beta, alpha]
#
# if self.treatment_fails:
#     self.Y.p = params[::-1]
#     self.X.p = self.X.p[::-1]


# return
#
# prior = self.prior
#
# a1, b1 = best_treated * prior, worst_treated * (1 - prior)
# a2, b2 = best_control * prior, worst_control * (1 - prior)
#
# c1 = agg_treated
# # c2 = agg_control
# c2 = best_control * prior + worst_control * (1 - prior) - agg_control
#
# det = a1 * b2 - a2 * b1
#
# treated_best = (b2 * c1 - b1 * c2) / det
# treated_worst = (a1 * c2 - a2 * c1) / det
#
# self.Z.p = self.prior
# self.X.p = [treated_worst, treated_best]
#
# if self.treatment_fails:
#     self.Y.p = params[::-1]
#     self.X.p = [treated_best, treated_worst]

