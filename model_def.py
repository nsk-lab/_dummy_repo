# model definition
class DummyModel(nn.Module):
	def __init__(self, n_class=11):
		super(DummyModel, self).__init__()
		self.fc0 = nn.Linear(3, n_class)

	def crop(self, x):
		x0 = x[0:4, 0:3, 46-32:46, 28 : 28+32]  # for test, 0:4 interpreted as 0:1
		x1 = x[0:4, 0:3, 46-32:46, 53 : 53+32]
		x2 = x[0:4, 0:3, 46-32:46, 91 : 91+32]
		x3 = x[0:4, 0:3, 46-32:46, 114: 114+32]
			
		return x0, x1, x2, x3

	def forward(self, x):
		x0, x1, x2, x3 = self.crop(x)
		o0 = self.forward_one(x0)
		o1 = self.forward_one(x1)
		o2 = self.forward_one(x2)
		o3 = self.forward_one(x3)
		return o0, o1, o2, o3

	def forward_one(self, x):
		"""assume input size is (32,32)"""
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = x.view(-1, 3)
		x = self.fc0(x)
		output = F.log_softmax(x, dim=1)
		return output
