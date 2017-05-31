local LinearWeightNorm, parent = torch.class('nn.SharedLinearWeightNorm', 'nn.Linear')

function LinearWeightNorm:__init(inputSize, outputSize, bias, eps)
    nn.Module.__init(self) -- Skip nn.Linear constructor

    local bias = ((bias == nil) and true) or bias

    self.eps = eps or 1e-16

    self.outputSize = outputSize
    self.inputSize = inputSize

    self.shared_v = torch.Tensor(outputSize, inputSize)
    self.shared_gradV = torch.Tensor(outputSize, inputSize)

    self.weight = torch.Tensor(outputSize, inputSize)

    self.shared_g = torch.Tensor(outputSize,1)
    self.shared_gradG = torch.Tensor(outputSize,1)

    self.shared_norm = torch.Tensor(outputSize,1)
    self.shared_scale = torch.Tensor(outputSize,1)

    if bias then
        self.bias = torch.Tensor(outputSize)
        self.gradBias = torch.Tensor(outputSize)
    end

    self:reset()
end

function LinearWeightNorm:evaluate()
    if self.train ~= false then
        self:updateWeightMatrix()
    end

    parent.evaluate(self)
end

function LinearWeightNorm:initFromWeight(weight)
    weight = weight or self.weight

    self.shared_g:norm(weight,2,2):clamp(self.eps,math.huge)
    self.shared_v:copy(weight)

    return self
end

function LinearWeightNorm.fromLinear(linear)
    local module = nn.SharedLinearWeightNorm(linear.weight:size(2), linear.weight:size(1), torch.isTensor(linear.bias))
    module.weight:copy(linear.weight)
    module:initFromWeight()

    if linear.bias then
        module.bias:copy(linear.bias)
    end

    return module
end

function LinearWeightNorm:toLinear()
    self:updateWeightMatrix()

    local module = nn.Linear(self.inputSize, self.outputSize, torch.isTensor(self.bias))

    module.weight:copy(self.weight)
    if self.bias then
        module.bias:copy(self.bias)
    end

    return module
end

function LinearWeightNorm:parameters()
    if self.bias then
        return {self.shared_v, self.shared_g, self.bias}, {self.shared_gradV, self.shared_gradG, self.gradBias}
    else
        return {self.shared_v, self.shared_g}, {self.shared_gradV, self.shared_gradG}
    end
end

function LinearWeightNorm:reset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1 / math.sqrt(self.inputSize)
    end
   
    self.weight:uniform(-stdv,stdv)
    self:initFromWeight()

    if self.bias then
        self.bias:uniform(-stdv,stdv)
    end
end

function LinearWeightNorm:updateWeightMatrix()
    if self.shared_norm:dim() == 0 then self.shared_norm:resizeAs(self.shared_g) end
    if self.shared_scale:dim() == 0 then self.shared_scale:resizeAs(self.shared_g) end
    if self.weight:dim() == 0 then self.weight:resizeAs(self.shared_v) end

    self.shared_norm:norm(self.shared_v,2,2):clamp(self.eps,math.huge)
    self.shared_scale:cdiv(self.shared_g,self.shared_norm)
    self.weight:cmul(self.shared_v,self.shared_scale:expandAs(self.shared_v))
end

function LinearWeightNorm:updateOutput(input)
    if self.train ~= false then
        self:updateWeightMatrix()
    end

    return parent.updateOutput(self, input)
end

function LinearWeightNorm:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    if input:dim() == 1 then
        self.shared_gradV:addr(scale, gradOutput, input)
        if self.bias then self.gradBias:add(scale, gradOutput) end
    elseif input:dim() == 2 then
        self.shared_gradV:addmm(scale, gradOutput:t(), input)
        if self.bias then
            -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
            self:updateAddBuffer(input)
            self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
        end
    end

    local scale = self.shared_scale:expandAs(self.shared_v)
    local norm = self.shared_norm:expandAs(self.shared_v)

    self.weight:cmul(self.shared_gradV,self.shared_v):cdiv(norm)
    self.shared_gradG:sum(self.weight,2)

    self.shared_gradV:cmul(scale)

    self.weight:cmul(self.shared_v,scale):cdiv(norm)
    self.weight:cmul(self.shared_gradG:expandAs(self.weight))

    self.shared_gradV:add(-1,self.weight)
end

function LinearWeightNorm:defaultAccUpdateGradParameters(input, gradOutput, lr)
    local shared_gradV = self.shared_gradV
    local shared_gradG = self.shared_gradG
    local gradBias = self.gradBias

    self.shared_gradV = self.shared_v
    self.shared_gradG = self.shared_g
    self.gradBias = self.bias

    self:accGradParameters(input, gradOutput, -lr)

    self.shared_gradV = shared_gradV
    self.shared_gradG = shared_gradG
    self.gradBias = gradBias
end

function LinearWeightNorm:clearState()
    nn.utils.clear(self, 'weight', 'shared_norm', 'shared_scale')
    return parent.clearState(self)
end

function LinearWeightNorm:__tostring__()
    return torch.type(self) ..
        string.format('(%d -> %d)', self.inputSize, self.outputSize) ..
        (self.bias == nil and ' without bias' or '')
end
