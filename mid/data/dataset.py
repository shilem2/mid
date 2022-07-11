from abc import ABC, abstractmethod

class Dataset(ABC):
	"""Dataset class, enables to get annotations, filter data, etc.
	"""

	@abstractmethod
	def get_study_id_list(self):
		pass

	@abstractmethod
	def get_ann(self):
		pass

	@staticmethod
	@abstractmethod
	def filter_anns_df():
		pass


