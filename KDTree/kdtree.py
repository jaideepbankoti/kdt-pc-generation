# main KDTree Class
class KDTree:
    def __init__(self, points, k=3):
        self.k = k
        # can set i to  be along dimension with largest spread
        self.kdt = self._make_kdt(points, 0)
        self.order = []
        
    def _make_kdt(self, points, i=0):
#         pdb.set_trace()
        if points.shape[0] > 1:
            points = points[points.argsort(axis=0)[:, i]]
            i  =  (i + 1) % self.k    # cycles through the dimension(or maybe could find the next best spread)
            partition  =  points.shape[0]  // 2    # by defaut partition points set at half and as per paper
            return [
                self._make_kdt(points[:partition], i),     # left node
                self._make_kdt(points[partition + 1:], i),     # right node
                points[partition],
            ]
        elif points.shape[0] == 1:
            return [
                None,
                None,
                points[0],
            ]
        
    def insert_point(self, point, i=0):
        dist = self.kdt[2][i] - point[i]
        i = (i +  1) % self.k
        for child, choice in ((0, dist >= 0), (1, dist < 0)):
            if choice and self.kdt[child] is  None:
                self.kdt[child] = [
                    None,
                    None,
                    point,
                ]
            elif choice:
                self.insert_point(point, i)
                
    def inorder(self, tree):
        if tree is not None:
            left, right, point = tree[0], tree[1], tree[2]
            self.inorder(left)
            self.order.append(point)
            self.inorder(right)