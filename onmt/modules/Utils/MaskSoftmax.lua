local MaskSoftMax, _ = torch.class('nn.MaskSoftMax', 'nn.Module')

function MaskSoftMax:updateOutput(input)
   local data = input[1] 
   local mask = input[2] -- should be in byte tensor
   
   data:maskedFill(mask, -9999999)
   
   data.THNN.SoftMax_updateOutput(
      data:cdata(),
      self.output:cdata()
   )
   return self.output
end

function MaskSoftMax:updateGradInput(input, gradOutput)
   local data = input[1]
   local mask = input[2]
   
   data:maskedFill(mask, -9999999)
   
   data.THNN.SoftMax_updateGradInput(
      data:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   
   -- mask doesn't have gradients
   if not self.dummy_out then
      self.dummy_out = mask:clone()
   end
   self.dummy_out:resizeAs(mask):zero()
   return {self.gradInput, self.dummy_out}
end
