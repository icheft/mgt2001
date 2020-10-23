import numpy as np
import html


def BayesTHM(pre_probs, event='D'):
    """
Output: (Bayes):
>>> [num1, num2] >> meaning that given event B has occurred, the probability that the previous event occurred is num1
============
Input:
>>> BayesTHM(np.array[[0.9, 0.1], # being the previous conditions
                          [0.99, 0.9], # being the conditional probabilities: ([P(B | A), P(B | A̅)])
                          [0.01, 0.1]]) # being the complementaries of the conditional probabilities: ([P(B̅ | A), P(B̅ | A̅)])

The deafult number of columns is 2.

Pass in `event='B'` to specify the post event you are to compare.
    """
    events = [event, html.unescape('{}&#773;'.format(event))]
    n = pre_probs.shape[1]
    m = pre_probs.shape[0] - 1

    # prior_probs is an array representing the previous probability
    # cond_probs is a matrix representing the previous condition probability

    # only the first row being G and G_c respectively
    prior_probs = pre_probs[0, :]
    cond_probs = pre_probs[1:, :]

    # joint_probs is a matrix representing the joint probability
    # p_c_sums is an array representing the probability of prior event
    # Bayes_M_G is a matrix representing the post conditional probability

    joint_probs = np.zeros((m, n))
    p_c_sums = np.zeros(n)
    bayes = np.zeros((m, n))

    for i in range(m):
        joint_probs[i, ] = prior_probs * cond_probs[i, ]
        p_c_sums[i] = np.sum(joint_probs[i, ])
        bayes[i, ] = joint_probs[i, ] / p_c_sums[i]
        to_print = """The probability of post event {i}:
{p_c_sum:.7f}
================
The Bayes probability given the post event {i}:
{bayes}
        """.format(i=events[i], p_c_sum=p_c_sums[i], bayes=bayes[i, ])
        print(to_print)
