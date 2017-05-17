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
local GlobalAttention, parent = torch.class('onmt.GlobalAttention', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function GlobalAttention:__init(dim, attType)
	self.attType = attType
	self.contextGate = contextGate
  parent.__init(self, self:_buildModel(dim, attType))
end

function GlobalAttention:_buildModel(dim, attType)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
	
  local targetT = nn.Linear(dim, dim, false)(inputs[1]) -- batchL x dim
  local context = inputs[2] -- batchL x sourceTimesteps x dim
  

  -- Get attention.
  local attn, softmaxAttn, alignmentVector
  if attType == 'general' then
		_G.logger:info(" * Using general type attention")
		attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
		attn = nn.Sum(3)(attn)
		softmaxAttn = nn.SoftMax()
		softmaxAttn.name = 'softmaxAttn'
		attn = softmaxAttn(attn)
		attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL
  elseif attType == 'mlp' then
		_G.logger:info(" * Using MLP type attention")
		local transformedContext = onmt.SequenceLinear(dim, dim, false)(context)
		local transformedHidden = nn.Replicate(1, 2)(targetT)
		local expands = nn.ExpandAs()({transformedContext, transformedHidden})
		
		-- neural network: W_h * H + W_c * C_t
		local sum = nn.CAddTable()(expands)
		local tanhSum = nn.Tanh()(sum)
		-- second layer of that network
		local score_ht_hs = onmt.SequenceLinear(dim, 1)(tanhSum)
		
		attn = nn.Sum(3)(score_ht_hs)
		local softmaxAttn = nn.SoftMax()
		softmaxAttn.name = 'softmaxAttn'
		attn = softmaxAttn(attn)
		alignmentVector = attn
		attn = nn.Replicate(1,2)(attn)
	else
		error('Attention type not implemented')
  end
	
  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  local contextVector = nn.Sum(2)(contextCombined)
	

  return nn.gModule(inputs, {contextVector})
end
