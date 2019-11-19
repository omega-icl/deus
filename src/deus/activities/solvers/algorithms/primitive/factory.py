from deus.activities.solvers.algorithms.primitive import \
    SamplingUniformlyOneBox, SamplingUniformlyOneEllipsoid
from deus.activities.solvers.algorithms.primitive import \
    SamplingUniformlyMultipleBodies
from deus.activities.solvers.algorithms.primitive import \
    KMeans, XMeans


class PrimitiveAlgorithmFactory:
    @classmethod
    def create(cls, name, settings=None):
        if name == "suob-box":
            return SamplingUniformlyOneBox()
        elif name == "suob-ellipsoid":
            return SamplingUniformlyOneEllipsoid()

        elif name == "sumb-mix":
            return SamplingUniformlyMultipleBodies()

        elif name == "clustering-kmeans":
            return KMeans(settings)
        elif name == "clustering-xmeans":
            return XMeans(settings)
