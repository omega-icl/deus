import abc

from deus.activities.solvers.algorithms import \
    PrimitiveAlgorithm
from deus.activities.solvers.algorithms.primitive.suob import \
    SamplingUniformlyOneBox, \
    SamplingUniformlyOneEllipsoid
from deus.activities.solvers.algorithms.primitive.sumb import \
    SamplingUniformlyMultipleBodies
from deus.activities.solvers.algorithms.primitive.clustering import \
    KMeans, XMeans
