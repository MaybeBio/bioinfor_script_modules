# 序列太长，无法embedding等，或者做其他处理，可以不断二分削减size
# 递归

	def split_frag( self, fragment ):
		"""
		Split fragments into half recursively, until all are < self.max_len.
		
		Input:
		----------
		fragment --> list of residue positions.

		Returns:
		----------
		fragment --> list of residue positions.
		"""
		if len( fragment ) <= self.max_len:
			return [fragment]
		else:
			shatter_point = len( fragment )// 2
			chunk1 = fragment[:shatter_point]
			chunk2 = fragment[shatter_point:]
			return self.split_frag( chunk1 ) + self.split_frag( chunk2 )
