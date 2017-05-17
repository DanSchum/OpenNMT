-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- nn.Reverse
--------------------------------------------------------------------------------

-- This module simply reverses an output in a dimension

local Reverse, parent = torch.class('nn.Reverse', 'nn.Module')

function Reverse:__init(dim, inplace)
    parent.__init(self)
    self.inplace = inplace or false
    self.dim = dim
end

function Reverse:updateOutput(input)
		local dim = self.dim
		local length = input:size(dim)

		if self.inplace then
			self.output = input
		else
			self.output = input:clone()
		end
		
		self.output = self.output:index(dim ,torch.linspace(length,1,length):long())
    
    return self.output
end

function Reverse:updateGradInput(input, gradOutput)
    
    local dim = self.dim
		local length = input:size(dim)
    
    -- just reverse the gradOutput
    self.gradInput = gradOutput:index(dim ,torch.linspace(length,1,length):long())
    
    return self.gradInput
end
