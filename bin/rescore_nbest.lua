require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('rescore_nbest.lua')

local options = {
  {'-src', '', [[Source sequence to decode (one line per sequence)]],
               {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-tgt', '', [[Nbest list for the target side]]},
  {'-output', 'pred.txt', [[Path to output the new n_best list]]}
}

cmd:setCmdLineOptions(options, 'Data')

onmt.translate.Rescorer.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

cmd:option('-time', false, [[Measure batch translation time]])

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.2f, " .. name .. " PPL: %.2f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  --~ local srcBatch = {}
  
  local hypReader = onmt.utils.FileReader.new(opt.tgt)
  

  local outFile = io.open(opt.output, 'w')

  local sentId = 0
  local batchId = 1

  local predScoreTotal = 0
  local predWordsTotal = 0
  --~ local goldScoreTotal = 0
  --~ local goldWordsTotal = 0

	-- First, dry-run to find the n-best list size
	local testReader = onmt.utils.FileReader.new(opt.tgt)
	
	local nbestSize = 0
	
	while true do
		local tgtTokens = testReader:next()
		--~ print(tgtTokens)	
		if #tgtTokens > 0 then
			nbestSize = nbestSize + 1
		else
			break
		end
	end
	
	_G.logger:info(" N-best size detected: " .. nbestSize)
	
	local rescorer = onmt.translate.Rescorer.new(opt, nbestSize)
	
	
  while true do
    local nbestList = {}
    local srcTokens = srcReader:next()
    -- end of file
    if srcTokens == nil then
			break
		end
    
    local currentHypId
    for n = 1, nbestSize do
			local currentTgtTokens = hypReader:next()
			currentHypId = currentTgtTokens[1]
			local currentHypScore = currentTgtTokens[#currentTgtTokens]
			
			local sentTokens = {}
			
			for i = 3, #currentTgtTokens - 2 do
				table.insert(sentTokens, currentTgtTokens[i])
			end
			
			local sentence = rescorer:buildInput(sentTokens)
			table.insert(nbestList, sentence)
    end
    
    hypReader:next() -- the empty line
    
    local srcSent = rescorer:buildInput(srcTokens)
    
    local results = rescorer:rescore(srcSent, nbestList)
    
    --~ print(results)
    
    for n = 1, nbestSize do
			local sentId = currentHypId
			local sentence = results[n].sent
			local score = results[n].score
			
			local sentWithScore = string.format("%i ||| %s ||| %.2f", sentId, sentence, score)
			
			outFile:write(sentWithScore .. '\n')
			_G.logger:info(sentWithScore)
    end
    
    outFile:write('\n')
    _G.logger:info('')
    
    
    
	

  end

  --~ if opt.time then
    --~ local time = timer:time()
    --~ local sentenceCount = sentId
    --~ _G.logger:info("Average sentence translation time (in seconds):\n")
    --~ _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    --~ _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    --~ _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  --~ end

  --~ reportScore('PRED', predScoreTotal, predWordsTotal)

  --~ if withGoldScore then
    --~ reportScore('GOLD', goldScoreTotal, goldWordsTotal)
  --~ end

  outFile:close()
  _G.logger:shutDown()
end

main()
