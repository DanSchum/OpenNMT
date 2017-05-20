local NilModule, _ = torch.class('nn.NilModule', 'nn.Module')

function NilModule:updateOutput(input)
   self.output = nil
   return self.output
end


function NilModule:updateGradInput(input, gradOutput)
   self.gradInput = nil
   return self.gradInput
end

function NilModule:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   return self
end
