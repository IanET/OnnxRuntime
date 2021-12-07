module OnnxRuntime

# Easier-to-use wrapper for LibOnnxRuntime
using ColorTypes
import LibOnnxRuntime as LOR
import Base.@kwdef

@kwdef mutable struct OrtGlobal
    apibase::LOR.OrtApiBase = LOR.OrtApiBase(C_NULL, C_NULL)
    api::LOR.OrtApi = LOR.OrtApi(ntuple(i -> UInt8(0), 1536))
end
const ORT = OrtGlobal()

function GetApi()   
    @ccall $(ORT.apibase.GetApi)(LOR.ORT_API_VERSION::UInt32)::Ptr{LOR.OrtApi}
end

function __init__()
    ORT.apibase = LOR.OrtGetApiBase() |> unsafe_load
    ORT.api = GetApi() |> unsafe_load
end 

struct OrtStatusError
    value::LOR.OrtStatusPtr
end

function checkStatus(pstatus::LOR.OrtStatusPtr)
    if (pstatus != P_ORT_STATUS_OK) throw(OrtStatusError(pstatus)) end
end

macro checkStatus(exp) 
    return :(checkStatus($(esc(exp)))) 
end

const POrtEnv = Ptr{LOR.OrtEnv}
const POrtStatus = Ptr{LOR.OrtStatus}
const POrtValue = Ptr{LOR.OrtValue}
const POrtSessionOptions = Ptr{LOR.OrtSessionOptions}
const POrtSession = Ptr{LOR.OrtSession}
const POrtMemoryInfo = Ptr{LOR.OrtMemoryInfo}
const POrtTensorTypeAndShapeInfo = Ptr{LOR.OrtTensorTypeAndShapeInfo}
const P_ORT_STATUS_OK = POrtStatus(0)

function createEnv(logid::String)
    renv = Ref{POrtEnv}(0)
    @checkStatus @ccall $(ORT.api.CreateEnv)(LOR.ORT_LOGGING_LEVEL_WARNING::Cint, logid::Cstring, renv::Ref{POrtEnv})::POrtStatus
    return renv[]
end

function createSessionOptions()
    rpsession_options = Ref(POrtSessionOptions(0))
    @checkStatus @ccall $(ORT.api.CreateSessionOptions)(rpsession_options::Ref{POrtSessionOptions})::POrtStatus
    return rpsession_options[]
end

function createSession(penv::POrtEnv, modelpath::String, poptions::POrtSessionOptions)
    rpsession = Ref(POrtSession(0))
    @checkStatus @ccall $(ORT.api.CreateSession)(penv::POrtEnv, modelpath::Cwstring, poptions::POrtSessionOptions, rpsession::Ref{POrtSession})::POrtStatus 
    return rpsession[]
end

function sessionGetInputCount(psession::POrtSession)
    rcount = Ref(Cuint(0))
    @checkStatus @ccall $(ORT.api.SessionGetInputCount)(psession::POrtSession, rcount::Ref{Cuint})::POrtStatus
    return rcount[]
end

function sessionGetOutputCount(psession::POrtSession)
    rcount = Ref(Cuint(0))
    @checkStatus @ccall $(ORT.api.SessionGetOutputCount)(psession::POrtSession, rcount::Ref{Cuint})::POrtStatus
    return rcount[]
end

function createCpuMemoryInfo()
    rpmemory_info = Ref(POrtMemoryInfo(0))
    @checkStatus @ccall $(ORT.api.CreateCpuMemoryInfo)(LOR.OrtArenaAllocator::Cint, LOR.OrtMemTypeDefault::Cint, rpmemory_info::Ref{POrtMemoryInfo})::POrtStatus
    return rpmemory_info[]
end

function createTensorWithDataAsOrtValue(pmemoryinfo::POrtMemoryInfo, input::Matrix{RGB{Float32}}, shape::Vector{Int64})
    input_len = sizeof(input)
    shape_len = length(shape)
    rinput_tensor = Ref(POrtValue(0))
    @checkStatus @ccall $(ORT.api.CreateTensorWithDataAsOrtValue)(
        pmemoryinfo::POrtMemoryInfo, 
        input::Ref{RGB{Float32}}, # Pass through
        input_len::Csize_t,
        shape::Ref{Int64},
        shape_len::Csize_t,
        LOR.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT::Cint,
        rinput_tensor::Ref{POrtValue}
        )::POrtStatus
    return rinput_tensor[]
end

function isTensor(tensor::POrtValue)
    ris_tensor = Ref(Cint(0))
    @checkStatus @ccall $(ORT.api.IsTensor)(tensor::POrtValue, ris_tensor::Ref{Cint})::POrtStatus
    return ris_tensor[] == 1 
end

toCharPtrs(sv::Vector{String}) = [pointer(transcode(UInt8, s)) for s in sv]

function run(psession::POrtSession, pinputTensor::POrtValue, inputNames::Vector{String}, outputNames::Vector{String})
    inputnameptrs = toCharPtrs(inputNames)
    outputnameptrs = toCharPtrs(outputNames)
    outputTensor = [POrtValue(0) for x in outputNames]
    rpinputTensor = Ref(pinputTensor)
    @checkStatus @ccall $(ORT.api.Run)(
        psession::POrtSession,
        C_NULL::Ref{Cvoid},
        inputnameptrs::Ptr{Cchar},
        rpinputTensor::Ref{POrtValue},
        length(inputNames)::Csize_t,
        outputnameptrs::Ptr{Cchar},
        length(outputNames)::Csize_t,
        outputTensor::Ref{POrtValue}
        )::POrtStatus
    return outputTensor
end

function getTensorTypeAndShape(tensor::POrtValue)
    rpshapenfo = Ref(POrtTensorTypeAndShapeInfo(0))
    @checkStatus @ccall $(ORT.api.GetTensorTypeAndShape)(tensor::POrtValue, rpshapenfo::Ref{POrtTensorTypeAndShapeInfo})::POrtStatus
    return rpshapenfo[]
end

function getDimensionsCount(pshapeinfo::POrtTensorTypeAndShapeInfo)
    rdim_count = Ref(Csize_t(0))
    @checkStatus @ccall $(ORT.api.GetDimensionsCount)(pshapeinfo::POrtTensorTypeAndShapeInfo, rdim_count::Ref{Csize_t})::POrtStatus
    return rdim_count[]
end

function getDimensions(pshapeinfo::POrtTensorTypeAndShapeInfo, dimcount::Csize_t)
    dims = zeros(Int64, dimcount)
    @checkStatus @ccall $(ORT.api.GetDimensions)(pshapeinfo::POrtTensorTypeAndShapeInfo, dims::Ref{Int64}, dimcount::Csize_t)::POrtStatus
    return dims
end

function getTensorMutableData(tensor::POrtValue)
    rpdata = Ref(Ptr{Float32}(0))
    @checkStatus @ccall $(ORT.api.GetTensorMutableData)(tensor::POrtValue, rpdata::Ref{Ptr{Float32}})::POrtStatus
    return rpdata[]
end

releaseMemoryInfo(meminfo::POrtMemoryInfo) = @ccall $(ORT.api.ReleaseMemoryInfo)(meminfo::POrtMemoryInfo)::Cvoid
releaseValue(pval::POrtValue) = @ccall $(ORT.api.ReleaseValue)(pval::POrtValue)::Cvoid
releaseSessionOptions(options::POrtSessionOptions) = @ccall $(ORT.api.ReleaseSessionOptions)(options::POrtSessionOptions)::Cvoid
releaseSession(session::POrtSession) = @ccall $(ORT.api.ReleaseSession)(session::POrtSession)::Cvoid
releaseEnv(env::POrtEnv) = @ccall $(ORT.api.ReleaseEnv)(env::POrtEnv)::Cvoid

end # module