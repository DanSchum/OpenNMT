require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('rescore_nbest.lua')

local options = {
  {'-src', '', [[Source sequence to decode (one line per sequence)]],
               {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-tgt', '', [[Nbest list for the target side]]},
  {'-output', 'pred.txt', [[Path to output the new n_best list]]},
  {'-batch_size', 128, [[Number of sentences to group for faster computatation]]}
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
  
	
	local rescorer = onmt.translate.Rescorer.new(opt, nbestSize)
	
	local hypID = -1
	local srcSent
	local tgtNBest = {}
	local tgtOldResults = {}
	
	while true do
		
		local currentTgtTokens = hypReader:next()
		
		-- end of file
    if currentTgtTokens == nil then
			local results = rescorer:rescore(srcSent, tgtNBest)
				
			for n = 1, #results do
				local sentId = hypID
				local sentence = results[n].sent
				local score = results[n].score
				
				
				local oldSent = table.concat(tgtTokens[n], ' ')
				local sentWithScore = string.format("%s %.2f", oldSent, score)
				
				outFile:write(sentWithScore .. '\n')
				_G.logger:info(sentWithScore)
			end
			
			outFile:write('\n')
			_G.logger:info('')
			-- proceed this batch then break
			break
		end
		
		local length = #currentTgtTokens
		
		if length > 0 then
			-- find the position of the last "|||"
			local lastMarker
			for j = length, 1, -1 do
				if currentTgtTokens[j] == '|||' then
					lastMarker = j
					break
				end
			end
			
			-- build sentence
			local sentTokens = {}
			for j = 3, lastMarker -1 do
				table.insert(sentTokens, currentTgtTokens[j])
			end
			
			for j = lastMarker + 1, length do
				local oldScores = {}
				table.insert(oldScores, currentTgtTokens[j])
			end
			
			local tgtHyp = rescorer:buildInput(sentTokens)
			
			local currentHypId = tonumber(currentTgtTokens[1])
			
			-- end of last sentence OR reach max batch size -> proceed this batch
			if currentHypId > hypID or #tgtNBest > opt.batch_size then
				
				if hypID > -1  then
					-- proceed this batch and print out result
					local results = rescorer:rescore(srcSent, tgtNBest)
					
					for n = 1, #results do
						local sentId = hypID
						local sentence = results[n].sent
						local score = results[n].score
						
						
						local oldSent = table.concat(tgtTokens[n], ' ')
						local sentWithScore = string.format("%s %.2f", oldSent, score)
						outFile:write(sentWithScore .. '\n')
						_G.logger:info(sentWithScore)
					end
					
					outFile:write('\n')
					_G.logger:info('')
				end
				
				if currentHypId > hypID then
					-- build next batch
					local srcTokens = srcReader:next()
					srcSent = rescorer:buildInput(srcTokens)
					tgtNBest = {tgtHyp}
					tgtTokens = {currentTgtTokens}
					
					if hypID == -1 then 
						hypID = currentHypId
					else
						hypID = hypID + 1
					end
					tgtOldResults = {oldScores}
				else
					-- we just reached the mini batch limit
					-- however our current sentenceID is not finished yet
					tgtNBest = {tgtHyp}
					tgtTokens = {currentTgtTokens}
					tgtOldResults = {oldScores}
				end
			else
				-- keep building the current mini batch
				table.insert(tgtNBest, tgtHyp)
				table.insert(tgtOldResults, oldScores)
				table.insert(tgtTokens, currentTgtTokens)
			end
		end
	end
    
  outFile:close()
  _G.logger:shutDown()
end

main()
