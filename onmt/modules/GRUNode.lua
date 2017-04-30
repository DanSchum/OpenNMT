local GRU, parent = torch.class('onmt.GRUNode', 'onmt.Network')

--]]
function GRU:__init(inputSize, hiddenSize, dropout)
	dropout = dropout or 0
  parent.__init(self, self:_buildModel(inputSize, hiddenSize, dropout))
end



function GRU:_buildModel(inputSize, hiddenSize, dropout)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  -- Recurrent input.
  local prevH = inputs[1]
  
  prevH = nn.Dropout(dropout)(prevH)
  
  -- Previous layer input.
  local x = inputs[2]
  
  -- compute 2 gates at once to get a little bit faster
  local i2h = nn.Linear(inputSize, 2 * hiddenSize)(x)
  local h2h = nn.Linear(hiddenSize, 2 * hiddenSize)(prevH)
  local allInputSums = nn.CAddTable()({i2h, h2h})
  local reshaped = nn.Reshape(2, hiddenSize)(allInputSums)
  local n1, n2 = nn.SplitTable(2)(reshaped):split(2)
    
  local uGate = nn.Sigmoid()(n1)
  
  local rGate = nn.Sigmoid()(n2)
  
  local gatedHidden = nn.CMulTable()({rGate, prevH})
  
  local p2 = nn.Linear(hiddenSize, hiddenSize)(gatedHidden)
  local p1 = nn.Linear(inputSize, hiddenSize)(x)
  
  local hiddenCandidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
  local zh = nn.CMulTable()({uGate, hiddenCandidate})
  local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(uGate)), prevH})
  
  local nextH = nn.CAddTable()({zh, zhm1})


  return nn.gModule(inputs, {nextH})
end

return GRU
