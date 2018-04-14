require 'optim'

-- modified to include a threshold for relative changes in the loss function as stopping criterion
local lbfgs_mod = require 'lbfgs'

---
--- MAIN FUNCTIONS
---

function runOptimization(params, net, content_image, content_losses, style_losses, temporal_losses,
    img, frameIdx, runIdx, max_iter, CSR)
  local isMultiPass = (runIdx ~= -1)

  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = max_iter,
      tolFunRelative = params.tol_loss_relative,
      tolFunRelativeInterval = params.tol_loss_relative_interval,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss, alwaysPrint)
    local should_print = (params.print_iter > 0 and t % params.print_iter == 0) or alwaysPrint
    if should_print then
      print(string.format('Iteration %d / %d', t, max_iter))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(temporal_losses) do
        print(string.format('  Temporal %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function print_end(t)
    --- calculate total loss
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(temporal_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    -- print informations
    maybe_print(t, loss, true)
  end

  local function maybe_save(t, isEnd)
    local should_save_intermed = params.save_iter > 0 and t % params.save_iter == 0
    local should_save_end = t == max_iter or isEnd
    if should_save_intermed or should_save_end then
      local filename = nil
      if isMultiPass then
        filename = build_OutFilename(params, frameIdx, runIdx)
      else
        filename = build_OutFilename(params, math.abs(frameIdx - params.start_number + 1), should_save_end and -1 or t)
      end
      save_image(img, filename)
    end
  end

  -- PRE-EVAL AFFINE DATA --
  local mean_pixel = torch.CudaTensor({103.939, 116.779, 123.68})
  local meanImage = mean_pixel:view(3, 1, 1):expandAs(content_image)
  -- END PRE-EVAL AFFINE DATA --

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this fucntion many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local best
  local function feval(x)
    num_calls = num_calls + 1

    -- AFFINE PRE-PROCESSING
    local output = torch.add(x, meanImage)
    local input  = torch.add(content_image, meanImage)
    -- END AFFINE PRE-PROCESSING

    net:forward(x)
    --local grad = net:backward(x, dy)

    -- ADD AFFINE CODE FROM PHOTO
    local grad = net:updateGradInput(x, dy)
    local c, h, w = content_image:size(1), content_image:size(2), content_image:size(3)
    local gradient_LocalAffine = MattingLaplacian(output, CSR, h, w):mul(params.lambda)

    if num_calls % params.save_iter_affine == 0 then
      best = SmoothLocalAffine(output, input, params.eps, params.patch, h, w, params.f_radius, params.f_edge)
    end

    grad = torch.add(grad, gradient_LocalAffine)
    -- END AFFINE CODE FROM PHOTO

    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(temporal_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss, false)
    -- Only need to print if single-pass algorithm is used.
    if not isMultiPass then
      maybe_save(num_calls, false)
    end

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  start_time = os.time()

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = lbfgs_mod.optimize(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, max_iter do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end

  end_time = os.time()
  elapsed_time = os.difftime(end_time-start_time)
  print("Running time: " .. elapsed_time .. "s")

  fn = params.serial .. '/best' .. tostring(frameIdx) .. tostring(params.index) .. '_t_' .. 'final' .. '.png'
  image.save(fn, best) -- Library image, ask to save.

  print_end(num_calls)
  maybe_save(num_calls, true)
end

-- Rebuild the network, insert style loss and return the indices for content and temporal loss
function buildNet(cnn, params, color_codes, style_image_caffe, color_content_masks, color_style_masks)

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")
  -- Which layer to use for the temporal loss. By default, it uses a pixel based loss, masked by the certainty
  --(indicated by initWeighted).
  local temporal_layers = params.temporal_weight > 0 and {'initWeighted'} or {}

  local style_losses = {}
  local contentLike_layers_indices = {}
  local contentLike_layers_type = {}

  local next_content_i, next_style_i, next_temporal_i = 1, 1, 1
  local current_layer_index = 1
  local net = nn.Sequential()

  -- Set up pixel based loss.
  if temporal_layers[next_temporal_i] == 'init' or temporal_layers[next_temporal_i] == 'initWeighted'  then
    print("Setting up temporal consistency.")
    table.insert(contentLike_layers_indices, current_layer_index)
    table.insert(contentLike_layers_type,
      (temporal_layers[next_temporal_i] == 'initWeighted') and 'prevPlusFlowWeighted' or 'prevPlusFlow')
    next_temporal_i = next_temporal_i + 1
  end

  -- Set up other loss modules.
  -- For content loss, only remember the indices at which they are inserted, because the content changes for each frame.
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    tv_mod = MaybePutOnGPU(tv_mod, params)
    net:add(tv_mod)
    current_layer_index = current_layer_index + 1
  end
  for i = 1, #cnn do
    if next_content_i <= #content_layers or next_style_i <= #style_layers or next_temporal_i <= #temporal_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      -- SPECIAL LAYER DETECTION FOR PHOTO
      local is_conv    = (layer_type == 'nn.SpatialConvolution' or layer_type == 'cudnn.SpatialConvolution')
      -- END
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        avg_pool_layer = MaybePutOnGPU(avg_pool_layer, params)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end

      -- SPECIAL CONTENT AND STYLE MASK PREPROCESSING PER CNN LAYER
      if is_pooling then
        for k = 1, #color_codes do
          color_content_masks[k] = image.scale(color_content_masks[k], math.ceil(color_content_masks[k]:size(2)/2), math.ceil(color_content_masks[k]:size(1)/2))
          color_style_masks[k]   = image.scale(color_style_masks[k],   math.ceil(color_style_masks[k]:size(2)/2),   math.ceil(color_style_masks[k]:size(1)/2))
        end
      elseif is_conv then
        local sap = nn.SpatialAveragePooling(3,3,1,1,1,1):float()
        for k = 1, #color_codes do
          color_content_masks[k] = sap:forward(color_content_masks[k]:repeatTensor(1,1,1))[1]:clone()
          color_style_masks[k]   = sap:forward(color_style_masks[k]:repeatTensor(1,1,1))[1]:clone()
        end
      end
      color_content_masks = deepcopy(color_content_masks)
      color_style_masks = deepcopy(color_style_masks)
      -- END SPECIAL PROCESSING

      current_layer_index = current_layer_index + 1
      if name == content_layers[next_content_i] then
        print("Setting up content layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'content')
        next_content_i = next_content_i + 1
      end
      if name == temporal_layers[next_temporal_i] then
        print("Setting up temporal layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'prevPlusFlow')
        next_temporal_i = next_temporal_i + 1
      end
      if name == style_layers[next_style_i] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        gram = MaybePutOnGPU(gram, params)
        local target = nil
        local target_features = net:forward(style_image_caffe):clone()
        local target_grams = {}

          -- local target_features = net:forward(style_image_caffe):clone()
          -- local target_i = gram:forward(target_features):clone()
          -- target_i:div(target_features:nElement())
          -- target_i:mul(style_blend_weights)
          -- target = target_i

          for j = 1, #color_codes do
            local l_style_mask_ori = color_style_masks[j]:clone():cuda()
            local l_style_mask = l_style_mask_ori:repeatTensor(1,1,1):expandAs(target_features)
            local l_style_mean = l_style_mask_ori:mean()

            local masked_target_features = torch.cmul(l_style_mask, target_features)
            local masked_target_gram = gram:forward(masked_target_features):clone()
            if l_style_mean > 0 then
              masked_target_gram:div(target_features:nElement() * l_style_mean)
            end
            table.insert(target_grams, masked_target_gram)
          end

        --local norm = params.normalize_gradients
        local loss_module = nn.StyleLossWithSeg(params.style_weight, target_grams, color_content_masks, color_codes, next_style_idx, false):float()
        loss_module = MaybePutOnGPU(loss_module, params)
        net:add(loss_module)
        current_layer_index = current_layer_index + 1
        table.insert(style_losses, loss_module)
        next_style_i = next_style_i + 1
      end
    end
  end
  return net, style_losses, contentLike_layers_indices, contentLike_layers_type
end

--
-- AFFINE MODULES FROM PHOTO
--
function MattingLaplacian(output, CSR, h, w)
  local N, c = CSR:size(1), CSR:size(2)
  local CSR_rowIdx = torch.CudaIntTensor(N):copy(torch.round(CSR[{{1,-1},1}]))
  local CSR_colIdx = torch.CudaIntTensor(N):copy(torch.round(CSR[{{1,-1},2}]))
  local CSR_val    = torch.CudaTensor(N):copy(CSR[{{1,-1},3}])

  local output01 = torch.div(output, 256.0)

  local grad = cuda_utils.matting_laplacian(output01, h, w, CSR_rowIdx, CSR_colIdx, CSR_val, N)

  grad:div(256.0)
  return grad
end

function SmoothLocalAffine(output, input, epsilon, patch, h, w, f_r, f_e)
  local output01 = torch.div(output, 256.0)
  local input01 = torch.div(input, 256.0)


  local filter_radius = f_r
  local sigma1, sigma2 = filter_radius / 3, f_e

  local best01= cuda_utils.smooth_local_affine(output01, input01, epsilon, patch, h, w, filter_radius, sigma1, sigma2)

  return best01
end

--
-- LOSS MODULES
--

-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Define an nn Module to compute content loss in-place
local WeightedContentLoss, parent = torch.class('nn.WeightedContentLoss', 'nn.Module')

function WeightedContentLoss:__init(strength, target, weights, normalize, loss_criterion)
  parent.__init(self)
  self.strength = strength
  if weights ~= nil then
    -- Take square root of the weights, because of the way the weights are applied
    -- to the mean square error function. We want w*(error^2), but we can only
    -- do (w*error)^2 = w^2 * error^2
    self.weights = torch.sqrt(weights)
    self.target = torch.cmul(target, self.weights)
  else
    self.target = target
    self.weights = nil
  end
  self.normalize = normalize or false
  self.loss = 0
  if loss_criterion == 'mse' then
    self.crit = nn.MSECriterion()
  elseif loss_criterion == 'smoothl1' then
    self.crit = nn.SmoothL1Criterion()
  else
    print('WARNING: Unknown flow loss criterion. Using MSE.')
    self.crit = nn.MSECriterion()
  end
end

function WeightedContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
    if self.weights ~= nil then
      self.loss = self.crit:forward(torch.cmul(input, self.weights), self.target) * self.strength
    else
      self.loss = self.crit:forward(input, self.target) * self.strength
    end
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function WeightedContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    if self.weights ~= nil then
      self.gradInput = self.crit:backward(torch.cmul(input, self.weights), self.target)
    else
      self.gradInput = self.crit:backward(input, self.target)
    end
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

-- Define style loss with segmentation
local StyleLossWithSeg, parent = torch.class('nn.StyleLossWithSeg', 'nn.Module')

--function StyleLossWithSeg:__init(strength, target_grams, color_content_masks, content_seg_idxs, layer_id)
function StyleLossWithSeg:__init(strength, target_grams, color_content_masks, color_codes, layer_id)
  parent.__init(self)
  self.strength = strength
  self.target_grams = target_grams
  self.color_content_masks = deepcopy(color_content_masks)
  self.color_codes = color_codes
  --self.content_seg_idxs = content_seg_idxs

  self.loss = 0
  self.gram = GramMatrix()
  self.crit = nn.MSECriterion()

  self.layer_id = layer_id
end

function StyleLossWithSeg:updateOutput(input)
  self.output = input
  return self.output
end

function StyleLossWithSeg:updateGradInput(input, gradOutput)
  self.loss = 0
  self.gradInput = gradOutput:clone()
  self.gradInput:zero()
  for j = 1, #self.color_codes do
    local l_content_mask_ori = self.color_content_masks[j]:clone():cuda()
    local l_content_mask = l_content_mask_ori:repeatTensor(1,1,1):expandAs(input)
    local l_content_mean = l_content_mask_ori:mean()

    local masked_input_features = torch.cmul(l_content_mask, input)
    local masked_input_gram = self.gram:forward(masked_input_features):clone()
    if l_content_mean > 0 then
      masked_input_gram:div(input:nElement() * l_content_mean)
    end

    local loss_j = self.crit:forward(masked_input_gram, self.target_grams[j])
    loss_j = loss_j * self.strength * l_content_mean
    self.loss = self.loss + loss_j

    local dG = self.crit:backward(masked_input_gram, self.target_grams[j])

    dG:div(input:nElement())

    local gradient = self.gram:backward(masked_input_features, dG)

    if self.normalize then
      gradient:div(torch.norm(gradient, 1) + 1e-8)
    end

    self.gradInput:add(gradient)
  end

  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0

  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function getContentLossModuleForLayer(net, layer_idx, target_img, params)
  local tmpNet = nn.Sequential()
  for i = 1, layer_idx-1 do
    local layer = net:get(i)
    tmpNet:add(layer)
  end
  local target = tmpNet:forward(target_img):clone()
  local loss_module = nn.ContentLoss(params.content_weight, target, params.normalize_gradients):float()
  loss_module = MaybePutOnGPU(loss_module, params)
  return loss_module
end

function getWeightedContentLossModuleForLayer(net, layer_idx, target_img, params, weights)
  local tmpNet = nn.Sequential()
  for i = 1, layer_idx-1 do
    local layer = net:get(i)
    tmpNet:add(layer)
  end
  local target = tmpNet:forward(target_img):clone()
  local loss_module = nn.WeightedContentLoss(params.temporal_weight, target, weights,
      params.normalize_gradients, params.temporal_loss_criterion):float()
  loss_module = MaybePutOnGPU(loss_module, params)
  return loss_module
end

---
--- HELPER FUNCTIONS
---

function MaybePutOnGPU(obj, params)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      return obj:cuda()
    else
      return obj:cl()
    end
  end
  return obj
end

-- COPIED FROM PHOTO STYLE (NO CONFLICTS)
function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-- COPIED FROM PHOT STYLE (NO CONFLICTS)
function ExtractMask(seg, color)
    local mask = nil

    if color == 'GREEN' then
      mask = torch.lt(seg[1], 0.15)
      mask:cmul(torch.gt(seg[2], 1-0.2))
      mask:cmul(torch.lt(seg[3], 0.15))
    elseif color == 'PURPLISH_RED' then
      mask = torch.cmul(torch.gt(seg[1], 0.5), torch.lt(seg[1], 0.5+0.3))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.2), torch.lt(seg[2], 0.4)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.3), torch.lt(seg[3], 0.55)))
    elseif color == 'ORANGE_YELLOW' then
      mask = torch.cmul(torch.gt(seg[1], 0.8), torch.lt(seg[1], 0.95))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.55), torch.lt(seg[2], 0.75)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.25), torch.lt(seg[3], 0.45)))
    elseif color == 'VIOLET' then
      mask = torch.cmul(torch.gt(seg[1], 0.3), torch.lt(seg[1], 0.5))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.3), torch.lt(seg[2], 0.4)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.4), torch.lt(seg[3], 0.7)))
    elseif color == 'PURPLISH_PINK' then
      mask = torch.cmul(torch.gt(seg[1], 0.75), torch.lt(seg[1], 0.95))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.5), torch.lt(seg[2], 0.7)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.55), torch.lt(seg[3], 0.8)))
    elseif color == 'BUFF' then
      mask = torch.gt(seg[1], 0.85)
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.75), torch.lt(seg[2], 0.95)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.4), torch.lt(seg[3], 0.6)))
    elseif color == 'ORANGE' then
      mask = torch.cmul(torch.gt(seg[1], 0.8), torch.lt(seg[1], 1))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.55), torch.lt(seg[2], 0.75)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.0), torch.lt(seg[3], 0.2)))
    elseif color == 'BROWN' then
      mask = torch.cmul(torch.gt(seg[1], 0.25), torch.lt(seg[1], 0.35))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.15), torch.lt(seg[2], 0.25)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.15), torch.lt(seg[3], 0.25)))
    elseif color == 'KHAKHI' then
      mask = torch.cmul(torch.gt(seg[1], 0.35), torch.lt(seg[1], 0.55))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.35), torch.lt(seg[2], 0.55)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.2), torch.lt(seg[3], 0.4)))
    elseif color == 'PINK' then
      mask = torch.cmul(torch.gt(seg[1], 0.6), torch.lt(seg[1], 0.8))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.35), torch.lt(seg[2], 0.55)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.35), torch.lt(seg[3], 0.55)))
    elseif color == 'BLACK' then
      mask = torch.lt(seg[1], 0.1)
      mask:cmul(torch.lt(seg[2], 0.1))
      mask:cmul(torch.lt(seg[3], 0.1))
    elseif color == 'WHITE' then
      mask = torch.gt(seg[1], 1-0.1)
      mask:cmul(torch.gt(seg[2], 1-0.1))
      mask:cmul(torch.gt(seg[3], 1-0.1))
    elseif color == 'RED' then
      mask = torch.gt(seg[1], 1-0.2)
      mask:cmul(torch.lt(seg[2], 0.15))
      mask:cmul(torch.lt(seg[3], 0.15))
    elseif color == 'BLUE' then
      mask = torch.lt(seg[1], 0.15)
      mask:cmul(torch.lt(seg[2], 0.15))
      mask:cmul(torch.gt(seg[3], 1-0.2))
    elseif color == 'YELLOW' then
      mask = torch.gt(seg[1], 1-0.2)
      mask:cmul(torch.gt(seg[2], 1-0.2))
      mask:cmul(torch.lt(seg[3], 0.15))
    elseif color == 'GREY' then
      mask = torch.cmul(torch.gt(seg[1], 0.5-0.1), torch.lt(seg[1], 0.5+0.1))
      mask:cmul(torch.cmul(torch.gt(seg[2], 0.5-0.1), torch.lt(seg[2], 0.5+0.1)))
      mask:cmul(torch.cmul(torch.gt(seg[3], 0.5-0.1), torch.lt(seg[3], 0.5+0.1)))
    elseif color == 'LIGHT_BLUE' then
      mask = torch.lt(seg[1], 0.15)
      mask:cmul(torch.gt(seg[2], 1-0.15))
      mask:cmul(torch.gt(seg[3], 1-0.15))
    elseif color == 'PURPLE' then
      mask = torch.gt(seg[1], 1-0.2)
      mask:cmul(torch.lt(seg[2], 0.15))
      mask:cmul(torch.gt(seg[3], 1-0.2))
    else
      print('ExtractMask(): color not recognized, color = ', color)
    end
    return mask:float()
end

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end

function save_image(img, fileName)
  local disp = deprocess(img:double())
  disp = image.minmax{tensor=disp, min=0, max=1}
  image.save(fileName, disp)
end

-- Checks whether a table contains a specific value
function tabl_contains(tabl, val)
   for i=1,#tabl do
      if tabl[i] == val then
         return true
      end
   end
   return false
end

-- Sums up all element in a given table
function tabl_sum(t)
  local sum = t[1]:clone()
  for i=2, #t do
    sum:add(t[i])
  end
  return sum
end

function str_split(str, delim, maxNb)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end
    if maxNb == nil or maxNb < 1 then
        maxNb = 0    -- No limit
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local nb = 1
    local lastPos
    for part, pos in string.gfind(str, pat) do
        result[nb] = part
        lastPos = pos
        nb = nb + 1
        if nb == maxNb then break end
    end
    -- Handle the last field
    result[nb] = string.sub(str, lastPos)
    return result
end

function fileExists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function calcNumberOfContentImages(params)
  local frameIdx = 1
  while frameIdx < 100000 do
    local fileName = string.format(params.content_pattern, frameIdx + params.start_number)
    if not fileExists(fileName) then return frameIdx end
    frameIdx = frameIdx + 1
  end
  -- If there are too many content frames, something may be wrong.
  return 0
end

function build_OutFilename(params, image_number, iterationOrRun)
  local ext = paths.extname(params.output_image)
  local basename = paths.basename(params.output_image, ext)
  local fileNameBase = '%s%s-' .. params.number_format
  if iterationOrRun == -1 then
    return string.format(fileNameBase .. '.%s',
      params.output_folder, basename, image_number, ext)
  else
    return string.format(fileNameBase .. '_%d.%s',
      params.output_folder, basename, image_number, iterationOrRun, ext)
  end
end

function getFormatedFlowFileName(pattern, fromIndex, toIndex)
  local flowFileName = pattern
  flowFileName = string.gsub(flowFileName, '{(.-)}',
    function(a) return string.format(a, fromIndex) end )
  flowFileName = string.gsub(flowFileName, '%[(.-)%]',
    function(a) return string.format(a, toIndex) end )
  return flowFileName
end

function getContentImage(frameIdx, params)
  local fileName = string.format(params.content_pattern, frameIdx)
  if not fileExists(fileName) then return nil end
  local content_image = image.load(string.format(params.content_pattern, frameIdx), 3)
  -- Load Segmentation --
  local segmentation_img = image.load(string.format(params.content_segmentation_path,frameIdx), 3)
  local img_scale = math.sqrt(content_image:size(2) * content_image:size(3) / (segmentation_img:size(3) * segmentation_img:size(2)))
  segmentation_img = image.scale(segmentation_img, segmentation_img:size(3) * img_scale, segmentation_img:size(2) * img_scale, 'bilinear')
  print("Content Segementation image size: " .. segmentation_img:size(3) .. " x " .. segmentation_img:size(2))
  segmentation_img = MaybePutOnGPU(segmentation_img, params)

  -- --- -- ---
  content_image = preprocess(content_image):float()
  content_image = MaybePutOnGPU(content_image, params)
  return content_image, segmentation_img
end

function getStyleImages(params)
  -- Needed to read content image size
  local firstContentImg = image.load(string.format(params.content_pattern, params.start_number), 3)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  -- Load Segmentation Image --
  local segmentation_img = image.load(params.style_segmentation_image, 3)
  local img_scale = math.sqrt(firstContentImg:size(2) * firstContentImg:size(3) / (segmentation_img:size(3) * segmentation_img:size(2)))
  segmentation_img = image.scale(segmentation_img, segmentation_img:size(3) * img_scale, segmentation_img:size(2) * img_scale, 'bilinear')
  print("Style Segementation image size: " .. segmentation_img:size(3) .. " x " .. segmentation_img:size(2))
  segmentation_img = MaybePutOnGPU(segmentation_img, params)
  --- ---- --- ---- --- ---
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
        * params.style_scale
    -- Scale the style image so that it's area equals the area of the content image multiplied by the style scale.
    local img_scale = math.sqrt(firstContentImg:size(2) * firstContentImg:size(3) / (img:size(3) * img:size(2)))
    img = image.scale(img, img:size(3) * img_scale, img:size(2) * img_scale, 'bilinear')
    print("Style image size: " .. img:size(3) .. " x " .. img:size(2))
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end
  for i = 1, #style_images_caffe do
     style_images_caffe[i] = MaybePutOnGPU(style_images_caffe[i], params)
  end
  return style_images_caffe,segmentation_img
end
