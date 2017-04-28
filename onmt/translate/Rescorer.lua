local Rescorer = torch.class('Rescorer')

local options = {
  {'-model', '', [[Path to model .t7 file]], {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-word_pen', 0, [[Word Penalty during decoding]]}
  }

local function clearStateModel(model)
  for _, submodule in pairs(model.modules) do
    if torch.type(submodule) == 'table' and submodule.modules then
      clearStateModel(submodule)
    else
      submodule:clearState()
      submodule:apply(function (m)
        nn.utils.clear(m, 'gradWeight', 'gradBias')
      end)
    end
  end
end


function Rescorer.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Rescorer')
end


function Rescorer:__init(args)
  self.opt = args
  onmt.utils.Cuda.init(self.opt)

  _G.logger:info('Loading \'' .. self.opt.model .. '\'...')
  self.checkpoint = torch.load(self.opt.model)
  _G.logger:info('Done...')
 

  self.models = {}
  self.models.encoder = onmt.Factory.loadEncoder(self.checkpoint.models.encoder)
  self.models.decoder = onmt.Factory.loadDecoder(self.checkpoint.models.decoder)
  
  clearStateModel(self.models.encoder)
	clearStateModel(self.models.decoder)

  self.models.encoder:evaluate()
  self.models.decoder:evaluate()

  onmt.utils.Cuda.convert(self.models.encoder)
  onmt.utils.Cuda.convert(self.models.decoder)

  self.dicts = self.checkpoint.dicts
  
  self.checkpoint = nil
  collectgarbage()
end

function Rescorer:buildInput(tokens)
  local words, features = onmt.utils.Features.extract(tokens)

  local data = {}
  data.words = words

  if #features > 0 then
    data.features = features
  end

  return data
end

function Rescorer:buildOutput(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), ' ')
end

function Rescorer:buildData(srcSent, nbestList)
  local srcData = {}
  srcData.words = {}
  srcData.features = {}

  local hypData
	hypData = {}
	hypData.words = {}
	hypData.features = {}
  
  local ignored = {}
  local indexMap = {}
  local index = 1
  
  -- we will build a minibatch of size nbest
  -- the source sentence will be duplicated 

  for b = 1, #nbestList do
  
		indexMap[index] = b
		index = index + 1
		
		table.insert(srcData.words, 
										self.dicts.src.words:convertToIdx(srcSent.words, onmt.Constants.UNK_WORD))
		if #self.dicts.src.features > 0 then
			table.insert(srcData.features,
                     onmt.utils.Features.generateSource(self.dicts.src.features, srcSent.features))
		end
		
		table.insert(hypData.words,
									self.dicts.tgt.words:convertToIdx(nbestList[b].words,
                                                       onmt.Constants.UNK_WORD,
                                                       onmt.Constants.BOS_WORD,
                                                       onmt.Constants.EOS_WORD))
    if #self.dicts.tgt.features > 0 then
          table.insert(hypData.features,
                       onmt.utils.Features.generateTarget(self.dicts.tgt.features, nbestList[b].features))
		end

  end

  return onmt.data.Dataset.new(srcData, hypData), ignored, indexMap
end

function Rescorer:buildTargetWords(pred, src, attn)
  local tokens = self.dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  if self.opt.replace_unk then
    for i = 1, #tokens do
      if tokens[i] == onmt.Constants.UNK_WORD then
        local _, maxIndex = attn[i]:max(1)
        local source = src[maxIndex[1]]

        if self.phraseTable and self.phraseTable:contains(source) then
          tokens[i] = self.phraseTable:lookup(source)
        else
          tokens[i] = source
        end
      end
    end
  end

  return tokens
end

function Rescorer:buildTargetFeatures(predFeats)

  local numFeatures = #predFeats[1]

  if numFeatures == 0 then
    return {}
  end

  local feats = {}
  for _ = 1, numFeatures do
    table.insert(feats, {})
  end

  for i = 2, #predFeats do
    for j = 1, numFeatures do
      table.insert(feats[j], self.dicts.tgt.features[j]:lookup(predFeats[i][j]))
    end
  end

  return feats
end

function Rescorer:rescoreBatch(batch)
  self.models.encoder:maskPadding()
  self.models.decoder:maskPadding()

  local encStates, context = self.models.encoder:forward(batch)
  
  hypScore = self.models.decoder:computeScore(batch, encStates, context)
  
  return hypScore
end

--[[ Translate a batch of source sequences.

Parameters:

  * `src` - a batch of tables containing:
    - `words`: the table of source words
    - `features`: the table of feaures sequences (`src.features[i][j]` is the value of the ith feature of the jth token)
  * `nbestlist` - a list of nbestlist to compute confidence score (same format as `src`)

Returns:

  * `results` - a batch of tables containing:
    - `goldScore`: if `gold` was given, this is the confidence score
    - `preds`: an array of `opt.n_best` tables containing:
      - `words`: the table of target words
      - `features`: the table of target features sequences
      - `attention`: the attention vectors of each target word over the source words
      - `score`: the confidence score of the prediction
]]
function Rescorer:rescore(src, nbestList)
  local data, ignored, indexMap = self:buildData(src, nbestList)

  local results = {}

  if data:batchCount() > 0 then
    local batch = data:getBatch()
    
    local hypScore = self:rescoreBatch(batch)
    
        
    for b = 1, batch.size do
			
			results[b] = {}
			
			results[b].score = hypScore[b]
			results[b].id = b
			results[b].sent = self:buildOutput(nbestList[b])
    end
		
  end

  return results
end

return Rescorer
