require('nngraph')

--[[ Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


    H_1 H_2 H_3 ... H_n
     q   q   q       q
      |  |   |       |
       \ |   |      /
           .....
         \   |  /
             a

Constructs a unit mapping:
  $$(H_1 .. H_n, q) => (a)$$
  Where H is of `batch x n x dim` and q is of `batch x dim`.

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

--]]
local GlobalMLPAttention, parent = torch.class('onmt.GlobalMLPAttention', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function GlobalMLPAttention:__init(dim)
  parent.__init(self, self:_buildModel(dim))
end

function GlobalMLPAttention:_buildModel(dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  --~ local targetT = nn.Linear(dim, dim, false)(inputs[1]) -- batchL x dim
  local ht = inputs[1]
  local context = inputs[2] -- batchL x sourceTimesteps x dim
  
  local transformedContext = onmt.SequenceLinear(dim, dim, false)(context)
  local transformedHidden = nn.Linear(dim, dim, false)(ht)
  transformedHidden = nn.Replicate(1, 2)(transformedHidden)
  
  local expands = nn.ExpandAs()({transformedContext, transformedHidden})
  
  local sum = nn.CAddTable()(expands)
  
	local tanhSum = nn.Tanh()(sum)
	
	local mapping = onmt.SequenceLinear(dim, 1)(tanhSum)
	local score_ht_hs = mapping


  -- Get attention.
  --~ local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
  attn = nn.Sum(3)(score_ht_hs)
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL

    -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  local contextVector = nn.Sum(2)(contextCombined) -- batchL x dim
  
  contextCombined = nn.JoinTable(2)({contextVector, inputs[1]})
  
  local contextGate = nn.Sigmoid()(nn.Linear(dim*2, dim, true)(contextCombined))
  local inputGate = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(contextGate))
  
  local gatedContext = nn.CMulTable()({contextGate, contextVector})
  local gatedInput   = nn.CMulTable()({inputGate, inputs[1]})
  
  local gatedContextCombined = nn.JoinTable(2)({gatedContext, gatedInput})
  
  --~ contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(gatedContextCombined))

  return nn.gModule(inputs, {contextOutput})
end
