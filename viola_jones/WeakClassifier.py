POSITIVE_CLASSIFICATION = 1
NEGATIVE_CLASSIFICATION = 0

class WeakClassifier():
  def __init__(self, feature, theta, p):
    # self.pos_regions = pos_regions;
    # self.feature = ([pos_regions], [neg_regions])
    self.feature = feature
    # self.neg_regions = neg_regions;
    self.theta = theta;
    self.p = p;
  

  def classify(self, integral_image):
    applied_feature = get_applied_feature(integral_image, self.feature)
    if self.p * applied_feature < self.p * self.theta:
      return POSITIVE_CLASSIFICATION
    return NEGATIVE_CLASSIFICATION
