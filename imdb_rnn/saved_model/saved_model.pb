°ä

Â""
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
>
DiagPart

input"T
diagonal"T"
Ttype:

2	
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
P
Qr

input"T
q"T
r"T"
full_matricesbool( "
Ttype:
2

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResourceGather
resource
indices"Tindices
output"dtype"
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Þ
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*1.13.12b'v1.13.1-2-g09e3b09e69'8º±	

embedding_inputPlaceholder*
dtype0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¯
5embedding/embeddings/Initializer/random_uniform/shapeConst*
valueB"ù  @   *'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
:
¡
3embedding/embeddings/Initializer/random_uniform/minConst*
valueB
 *ÍÌL½*'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
: 
¡
3embedding/embeddings/Initializer/random_uniform/maxConst*
valueB
 *ÍÌL=*'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
: 
å
=embedding/embeddings/Initializer/random_uniform/RandomUniformRandomUniform5embedding/embeddings/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	ù?@*
T0*'
_class
loc:@embedding/embeddings
î
3embedding/embeddings/Initializer/random_uniform/subSub3embedding/embeddings/Initializer/random_uniform/max3embedding/embeddings/Initializer/random_uniform/min*
T0*'
_class
loc:@embedding/embeddings*
_output_shapes
: 

3embedding/embeddings/Initializer/random_uniform/mulMul=embedding/embeddings/Initializer/random_uniform/RandomUniform3embedding/embeddings/Initializer/random_uniform/sub*'
_class
loc:@embedding/embeddings*
_output_shapes
:	ù?@*
T0
ó
/embedding/embeddings/Initializer/random_uniformAdd3embedding/embeddings/Initializer/random_uniform/mul3embedding/embeddings/Initializer/random_uniform/min*
T0*'
_class
loc:@embedding/embeddings*
_output_shapes
:	ù?@
®
embedding/embeddingsVarHandleOp*
shape:	ù?@*%
shared_nameembedding/embeddings*'
_class
loc:@embedding/embeddings*
dtype0*
_output_shapes
: 
y
5embedding/embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOpembedding/embeddings*
_output_shapes
: 
¬
embedding/embeddings/AssignAssignVariableOpembedding/embeddings/embedding/embeddings/Initializer/random_uniform*'
_class
loc:@embedding/embeddings*
dtype0
§
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
dtype0*
_output_shapes
:	ù?@*'
_class
loc:@embedding/embeddings
q
embedding/CastCastembedding_input*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*

DstT0
Î
embedding/embedding_lookupResourceGatherembedding/embeddingsembedding/Cast*
Tindices0*'
_class
loc:@embedding/embeddings*
dtype0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
³
#embedding/embedding_lookup/IdentityIdentityembedding/embedding_lookup*
T0*'
_class
loc:@embedding/embeddings*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@

%embedding/embedding_lookup/Identity_1Identity#embedding/embedding_lookup/Identity*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
É
Bbidirectional/forward_lstm/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"@      *4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
dtype0
»
@bidirectional/forward_lstm/kernel/Initializer/random_uniform/minConst*
valueB
 *7¾*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
dtype0*
_output_shapes
: 
»
@bidirectional/forward_lstm/kernel/Initializer/random_uniform/maxConst*
valueB
 *7>*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
dtype0*
_output_shapes
: 

Jbidirectional/forward_lstm/kernel/Initializer/random_uniform/RandomUniformRandomUniformBbidirectional/forward_lstm/kernel/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
dtype0*
_output_shapes
:	@
¢
@bidirectional/forward_lstm/kernel/Initializer/random_uniform/subSub@bidirectional/forward_lstm/kernel/Initializer/random_uniform/max@bidirectional/forward_lstm/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
_output_shapes
: 
µ
@bidirectional/forward_lstm/kernel/Initializer/random_uniform/mulMulJbidirectional/forward_lstm/kernel/Initializer/random_uniform/RandomUniform@bidirectional/forward_lstm/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
_output_shapes
:	@
§
<bidirectional/forward_lstm/kernel/Initializer/random_uniformAdd@bidirectional/forward_lstm/kernel/Initializer/random_uniform/mul@bidirectional/forward_lstm/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
_output_shapes
:	@
Õ
!bidirectional/forward_lstm/kernelVarHandleOp*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
dtype0*
_output_shapes
: *
shape:	@*2
shared_name#!bidirectional/forward_lstm/kernel

Bbidirectional/forward_lstm/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!bidirectional/forward_lstm/kernel*
_output_shapes
: 
à
(bidirectional/forward_lstm/kernel/AssignAssignVariableOp!bidirectional/forward_lstm/kernel<bidirectional/forward_lstm/kernel/Initializer/random_uniform*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
dtype0
Î
5bidirectional/forward_lstm/kernel/Read/ReadVariableOpReadVariableOp!bidirectional/forward_lstm/kernel*
_output_shapes
:	@*4
_class*
(&loc:@bidirectional/forward_lstm/kernel*
dtype0
Ü
Kbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/shapeConst*
valueB"   @   *>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:
Ï
Jbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/meanConst*
valueB
 *    *>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
Ñ
Lbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/stddevConst*
valueB
 *  ?*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
¶
Zbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalKbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/shape*
T0*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
ä
Ibidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/mulMulZbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormalLbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/stddev*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
:	@*
T0
Í
Ebidirectional/forward_lstm/recurrent_kernel/Initializer/random_normalAddIbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/mulJbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal/mean*
_output_shapes
:	@*
T0*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel
û
:bidirectional/forward_lstm/recurrent_kernel/Initializer/QrQrEbidirectional/forward_lstm/recurrent_kernel/Initializer/random_normal*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*)
_output_shapes
:	@:@@*
T0
ï
@bidirectional/forward_lstm/recurrent_kernel/Initializer/DiagPartDiagPart<bidirectional/forward_lstm/recurrent_kernel/Initializer/Qr:1*
_output_shapes
:@*
T0*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel
ë
<bidirectional/forward_lstm/recurrent_kernel/Initializer/SignSign@bidirectional/forward_lstm/recurrent_kernel/Initializer/DiagPart*
T0*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
:@
¦
;bidirectional/forward_lstm/recurrent_kernel/Initializer/mulMul:bidirectional/forward_lstm/recurrent_kernel/Initializer/Qr<bidirectional/forward_lstm/recurrent_kernel/Initializer/Sign*
T0*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
:	@
è
Wbidirectional/forward_lstm/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*
valueB"       *>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:
ß
Rbidirectional/forward_lstm/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose;bidirectional/forward_lstm/recurrent_kernel/Initializer/mulWbidirectional/forward_lstm/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
:	@*
T0
Ö
Ebidirectional/forward_lstm/recurrent_kernel/Initializer/Reshape/shapeConst*
valueB"@      *>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:
Ï
?bidirectional/forward_lstm/recurrent_kernel/Initializer/ReshapeReshapeRbidirectional/forward_lstm/recurrent_kernel/Initializer/matrix_transpose/transposeEbidirectional/forward_lstm/recurrent_kernel/Initializer/Reshape/shape*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
:	@*
T0
Ä
?bidirectional/forward_lstm/recurrent_kernel/Initializer/mul_1/xConst*
valueB
 *  ?*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
°
=bidirectional/forward_lstm/recurrent_kernel/Initializer/mul_1Mul?bidirectional/forward_lstm/recurrent_kernel/Initializer/mul_1/x?bidirectional/forward_lstm/recurrent_kernel/Initializer/Reshape*
T0*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
:	@
ó
+bidirectional/forward_lstm/recurrent_kernelVarHandleOp*
shape:	@*<
shared_name-+bidirectional/forward_lstm/recurrent_kernel*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
§
Lbidirectional/forward_lstm/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
: 
ÿ
2bidirectional/forward_lstm/recurrent_kernel/AssignAssignVariableOp+bidirectional/forward_lstm/recurrent_kernel=bidirectional/forward_lstm/recurrent_kernel/Initializer/mul_1*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0
ì
?bidirectional/forward_lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOp+bidirectional/forward_lstm/recurrent_kernel*>
_class4
20loc:@bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
²
1bidirectional/forward_lstm/bias/Initializer/zerosConst*
valueB@*    *2
_class(
&$loc:@bidirectional/forward_lstm/bias*
dtype0*
_output_shapes
:@
±
0bidirectional/forward_lstm/bias/Initializer/onesConst*
valueB@*  ?*2
_class(
&$loc:@bidirectional/forward_lstm/bias*
dtype0*
_output_shapes
:@
¶
3bidirectional/forward_lstm/bias/Initializer/zeros_1Const*
valueB*    *2
_class(
&$loc:@bidirectional/forward_lstm/bias*
dtype0*
_output_shapes	
:
­
7bidirectional/forward_lstm/bias/Initializer/concat/axisConst*
value	B : *2
_class(
&$loc:@bidirectional/forward_lstm/bias*
dtype0*
_output_shapes
: 
ô
2bidirectional/forward_lstm/bias/Initializer/concatConcatV21bidirectional/forward_lstm/bias/Initializer/zeros0bidirectional/forward_lstm/bias/Initializer/ones3bidirectional/forward_lstm/bias/Initializer/zeros_17bidirectional/forward_lstm/bias/Initializer/concat/axis*
T0*2
_class(
&$loc:@bidirectional/forward_lstm/bias*
N*
_output_shapes	
:
Ë
bidirectional/forward_lstm/biasVarHandleOp*
_output_shapes
: *
shape:*0
shared_name!bidirectional/forward_lstm/bias*2
_class(
&$loc:@bidirectional/forward_lstm/bias*
dtype0

@bidirectional/forward_lstm/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpbidirectional/forward_lstm/bias*
_output_shapes
: 
Ð
&bidirectional/forward_lstm/bias/AssignAssignVariableOpbidirectional/forward_lstm/bias2bidirectional/forward_lstm/bias/Initializer/concat*2
_class(
&$loc:@bidirectional/forward_lstm/bias*
dtype0
Ä
3bidirectional/forward_lstm/bias/Read/ReadVariableOpReadVariableOpbidirectional/forward_lstm/bias*2
_class(
&$loc:@bidirectional/forward_lstm/bias*
dtype0*
_output_shapes	
:
Ë
Cbidirectional/backward_lstm/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
:
½
Abidirectional/backward_lstm/kernel/Initializer/random_uniform/minConst*
valueB
 *7¾*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
: 
½
Abidirectional/backward_lstm/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *7>*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
dtype0

Kbidirectional/backward_lstm/kernel/Initializer/random_uniform/RandomUniformRandomUniformCbidirectional/backward_lstm/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	@*
T0*5
_class+
)'loc:@bidirectional/backward_lstm/kernel
¦
Abidirectional/backward_lstm/kernel/Initializer/random_uniform/subSubAbidirectional/backward_lstm/kernel/Initializer/random_uniform/maxAbidirectional/backward_lstm/kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
_output_shapes
: 
¹
Abidirectional/backward_lstm/kernel/Initializer/random_uniform/mulMulKbidirectional/backward_lstm/kernel/Initializer/random_uniform/RandomUniformAbidirectional/backward_lstm/kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
_output_shapes
:	@
«
=bidirectional/backward_lstm/kernel/Initializer/random_uniformAddAbidirectional/backward_lstm/kernel/Initializer/random_uniform/mulAbidirectional/backward_lstm/kernel/Initializer/random_uniform/min*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
_output_shapes
:	@*
T0
Ø
"bidirectional/backward_lstm/kernelVarHandleOp*3
shared_name$"bidirectional/backward_lstm/kernel*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
: *
shape:	@

Cbidirectional/backward_lstm/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"bidirectional/backward_lstm/kernel*
_output_shapes
: 
ä
)bidirectional/backward_lstm/kernel/AssignAssignVariableOp"bidirectional/backward_lstm/kernel=bidirectional/backward_lstm/kernel/Initializer/random_uniform*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
dtype0
Ñ
6bidirectional/backward_lstm/kernel/Read/ReadVariableOpReadVariableOp"bidirectional/backward_lstm/kernel*5
_class+
)'loc:@bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
:	@
Þ
Lbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/shapeConst*
_output_shapes
:*
valueB"   @   *?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0
Ñ
Kbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/meanConst*
valueB
 *    *?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
Ó
Mbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/stddevConst*
valueB
 *  ?*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
¹
[bidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormalRandomStandardNormalLbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/shape*
T0*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
è
Jbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/mulMul[bidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/RandomStandardNormalMbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/stddev*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
:	@*
T0
Ñ
Fbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normalAddJbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/mulKbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal/mean*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
:	@*
T0
þ
;bidirectional/backward_lstm/recurrent_kernel/Initializer/QrQrFbidirectional/backward_lstm/recurrent_kernel/Initializer/random_normal*
T0*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*)
_output_shapes
:	@:@@
ò
Abidirectional/backward_lstm/recurrent_kernel/Initializer/DiagPartDiagPart=bidirectional/backward_lstm/recurrent_kernel/Initializer/Qr:1*
_output_shapes
:@*
T0*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel
î
=bidirectional/backward_lstm/recurrent_kernel/Initializer/SignSignAbidirectional/backward_lstm/recurrent_kernel/Initializer/DiagPart*
T0*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
:@
ª
<bidirectional/backward_lstm/recurrent_kernel/Initializer/mulMul;bidirectional/backward_lstm/recurrent_kernel/Initializer/Qr=bidirectional/backward_lstm/recurrent_kernel/Initializer/Sign*
T0*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
:	@
ê
Xbidirectional/backward_lstm/recurrent_kernel/Initializer/matrix_transpose/transpose/permConst*
_output_shapes
:*
valueB"       *?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0
ã
Sbidirectional/backward_lstm/recurrent_kernel/Initializer/matrix_transpose/transpose	Transpose<bidirectional/backward_lstm/recurrent_kernel/Initializer/mulXbidirectional/backward_lstm/recurrent_kernel/Initializer/matrix_transpose/transpose/perm*
_output_shapes
:	@*
T0*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel
Ø
Fbidirectional/backward_lstm/recurrent_kernel/Initializer/Reshape/shapeConst*
valueB"@      *?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:
Ó
@bidirectional/backward_lstm/recurrent_kernel/Initializer/ReshapeReshapeSbidirectional/backward_lstm/recurrent_kernel/Initializer/matrix_transpose/transposeFbidirectional/backward_lstm/recurrent_kernel/Initializer/Reshape/shape*
_output_shapes
:	@*
T0*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel
Æ
@bidirectional/backward_lstm/recurrent_kernel/Initializer/mul_1/xConst*
valueB
 *  ?*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
: 
´
>bidirectional/backward_lstm/recurrent_kernel/Initializer/mul_1Mul@bidirectional/backward_lstm/recurrent_kernel/Initializer/mul_1/x@bidirectional/backward_lstm/recurrent_kernel/Initializer/Reshape*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
:	@*
T0
ö
,bidirectional/backward_lstm/recurrent_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@*=
shared_name.,bidirectional/backward_lstm/recurrent_kernel*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel
©
Mbidirectional/backward_lstm/recurrent_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp,bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
: 

3bidirectional/backward_lstm/recurrent_kernel/AssignAssignVariableOp,bidirectional/backward_lstm/recurrent_kernel>bidirectional/backward_lstm/recurrent_kernel/Initializer/mul_1*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel*
dtype0
ï
@bidirectional/backward_lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOp,bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@*?
_class5
31loc:@bidirectional/backward_lstm/recurrent_kernel
´
2bidirectional/backward_lstm/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *3
_class)
'%loc:@bidirectional/backward_lstm/bias
³
1bidirectional/backward_lstm/bias/Initializer/onesConst*
valueB@*  ?*3
_class)
'%loc:@bidirectional/backward_lstm/bias*
dtype0*
_output_shapes
:@
¸
4bidirectional/backward_lstm/bias/Initializer/zeros_1Const*
dtype0*
_output_shapes	
:*
valueB*    *3
_class)
'%loc:@bidirectional/backward_lstm/bias
¯
8bidirectional/backward_lstm/bias/Initializer/concat/axisConst*
value	B : *3
_class)
'%loc:@bidirectional/backward_lstm/bias*
dtype0*
_output_shapes
: 
ú
3bidirectional/backward_lstm/bias/Initializer/concatConcatV22bidirectional/backward_lstm/bias/Initializer/zeros1bidirectional/backward_lstm/bias/Initializer/ones4bidirectional/backward_lstm/bias/Initializer/zeros_18bidirectional/backward_lstm/bias/Initializer/concat/axis*
T0*3
_class)
'%loc:@bidirectional/backward_lstm/bias*
N*
_output_shapes	
:
Î
 bidirectional/backward_lstm/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*1
shared_name" bidirectional/backward_lstm/bias*3
_class)
'%loc:@bidirectional/backward_lstm/bias

Abidirectional/backward_lstm/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp bidirectional/backward_lstm/bias*
_output_shapes
: 
Ô
'bidirectional/backward_lstm/bias/AssignAssignVariableOp bidirectional/backward_lstm/bias3bidirectional/backward_lstm/bias/Initializer/concat*3
_class)
'%loc:@bidirectional/backward_lstm/bias*
dtype0
Ç
4bidirectional/backward_lstm/bias/Read/ReadVariableOpReadVariableOp bidirectional/backward_lstm/bias*3
_class)
'%loc:@bidirectional/backward_lstm/bias*
dtype0*
_output_shapes	
:
h
bidirectional/ShapeShape%embedding/embedding_lookup/Identity_1*
_output_shapes
:*
T0
k
!bidirectional/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
m
#bidirectional/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#bidirectional/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ó
bidirectional/strided_sliceStridedSlicebidirectional/Shape!bidirectional/strided_slice/stack#bidirectional/strided_slice/stack_1#bidirectional/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
[
bidirectional/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: 
w
bidirectional/zeros/mulMulbidirectional/strided_slicebidirectional/zeros/mul/y*
_output_shapes
: *
T0
]
bidirectional/zeros/Less/yConst*
value
B :è*
dtype0*
_output_shapes
: 
v
bidirectional/zeros/LessLessbidirectional/zeros/mulbidirectional/zeros/Less/y*
_output_shapes
: *
T0
^
bidirectional/zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: 

bidirectional/zeros/packedPackbidirectional/strided_slicebidirectional/zeros/packed/1*
T0*
N*
_output_shapes
:
^
bidirectional/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/zerosFillbidirectional/zeros/packedbidirectional/zeros/Const*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
]
bidirectional/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: 
{
bidirectional/zeros_1/mulMulbidirectional/strided_slicebidirectional/zeros_1/mul/y*
T0*
_output_shapes
: 
_
bidirectional/zeros_1/Less/yConst*
value
B :è*
dtype0*
_output_shapes
: 
|
bidirectional/zeros_1/LessLessbidirectional/zeros_1/mulbidirectional/zeros_1/Less/y*
_output_shapes
: *
T0
`
bidirectional/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: 

bidirectional/zeros_1/packedPackbidirectional/strided_slicebidirectional/zeros_1/packed/1*
T0*
N*
_output_shapes
:
`
bidirectional/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

bidirectional/zeros_1Fillbidirectional/zeros_1/packedbidirectional/zeros_1/Const*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
q
bidirectional/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
¨
bidirectional/transpose	Transpose%embedding/embedding_lookup/Identity_1bidirectional/transpose/perm*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0
\
bidirectional/Shape_1Shapebidirectional/transpose*
T0*
_output_shapes
:
m
#bidirectional/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
o
%bidirectional/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ý
bidirectional/strided_slice_1StridedSlicebidirectional/Shape_1#bidirectional/strided_slice_1/stack%bidirectional/strided_slice_1/stack_1%bidirectional/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
·
bidirectional/TensorArrayTensorArrayV3bidirectional/strided_slice_1*
identical_element_shapes(*!
tensor_array_name
input_ta_0*
dtype0*
_output_shapes

:: 
m
&bidirectional/TensorArrayUnstack/ShapeShapebidirectional/transpose*
_output_shapes
:*
T0
~
4bidirectional/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

6bidirectional/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

6bidirectional/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ò
.bidirectional/TensorArrayUnstack/strided_sliceStridedSlice&bidirectional/TensorArrayUnstack/Shape4bidirectional/TensorArrayUnstack/strided_slice/stack6bidirectional/TensorArrayUnstack/strided_slice/stack_16bidirectional/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
n
,bidirectional/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
n
,bidirectional/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
à
&bidirectional/TensorArrayUnstack/rangeRange,bidirectional/TensorArrayUnstack/range/start.bidirectional/TensorArrayUnstack/strided_slice,bidirectional/TensorArrayUnstack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
Hbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3bidirectional/TensorArray&bidirectional/TensorArrayUnstack/rangebidirectional/transposebidirectional/TensorArray:1*
T0**
_class 
loc:@bidirectional/transpose*
_output_shapes
: 
m
#bidirectional/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional/strided_slice_2StridedSlicebidirectional/transpose#bidirectional/strided_slice_2/stack%bidirectional/strided_slice_2/stack_1%bidirectional/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOpReadVariableOp!bidirectional/forward_lstm/kernel*
dtype0*
_output_shapes
:	@
t
#bidirectional/strided_slice_3/stackConst*
valueB"        *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_3/stack_1Const*
_output_shapes
:*
valueB"    @   *
dtype0
v
%bidirectional/strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_3StridedSlicebidirectional/ReadVariableOp#bidirectional/strided_slice_3/stack%bidirectional/strided_slice_3/stack_1%bidirectional/strided_slice_3/stack_2*

begin_mask*
T0*
Index0*
end_mask*
_output_shapes

:@@

bidirectional/MatMulMatMulbidirectional/strided_slice_2bidirectional/strided_slice_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_1ReadVariableOp!bidirectional/forward_lstm/kernel*
dtype0*
_output_shapes
:	@
t
#bidirectional/strided_slice_4/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_4/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_4StridedSlicebidirectional/ReadVariableOp_1#bidirectional/strided_slice_4/stack%bidirectional/strided_slice_4/stack_1%bidirectional/strided_slice_4/stack_2*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@

bidirectional/MatMul_1MatMulbidirectional/strided_slice_2bidirectional/strided_slice_4*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/ReadVariableOp_2ReadVariableOp!bidirectional/forward_lstm/kernel*
_output_shapes
:	@*
dtype0
t
#bidirectional/strided_slice_5/stackConst*
valueB"       *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_5/stack_1Const*
valueB"    À   *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      

bidirectional/strided_slice_5StridedSlicebidirectional/ReadVariableOp_2#bidirectional/strided_slice_5/stack%bidirectional/strided_slice_5/stack_1%bidirectional/strided_slice_5/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:@@

bidirectional/MatMul_2MatMulbidirectional/strided_slice_2bidirectional/strided_slice_5*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/ReadVariableOp_3ReadVariableOp!bidirectional/forward_lstm/kernel*
dtype0*
_output_shapes
:	@
t
#bidirectional/strided_slice_6/stackConst*
valueB"    À   *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_6/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
v
%bidirectional/strided_slice_6/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_6StridedSlicebidirectional/ReadVariableOp_3#bidirectional/strided_slice_6/stack%bidirectional/strided_slice_6/stack_1%bidirectional/strided_slice_6/stack_2*
_output_shapes

:@@*
T0*
Index0*

begin_mask*
end_mask

bidirectional/MatMul_3MatMulbidirectional/strided_slice_2bidirectional/strided_slice_6*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
{
bidirectional/ReadVariableOp_4ReadVariableOpbidirectional/forward_lstm/bias*
dtype0*
_output_shapes	
:
m
#bidirectional/strided_slice_7/stackConst*
dtype0*
_output_shapes
:*
valueB: 
o
%bidirectional/strided_slice_7/stack_1Const*
_output_shapes
:*
valueB:@*
dtype0
o
%bidirectional/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional/strided_slice_7StridedSlicebidirectional/ReadVariableOp_4#bidirectional/strided_slice_7/stack%bidirectional/strided_slice_7/stack_1%bidirectional/strided_slice_7/stack_2*

begin_mask*
T0*
Index0*
_output_shapes
:@

bidirectional/BiasAddBiasAddbidirectional/MatMulbidirectional/strided_slice_7*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
{
bidirectional/ReadVariableOp_5ReadVariableOpbidirectional/forward_lstm/bias*
dtype0*
_output_shapes	
:
m
#bidirectional/strided_slice_8/stackConst*
valueB:@*
dtype0*
_output_shapes
:
p
%bidirectional/strided_slice_8/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%bidirectional/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ò
bidirectional/strided_slice_8StridedSlicebidirectional/ReadVariableOp_5#bidirectional/strided_slice_8/stack%bidirectional/strided_slice_8/stack_1%bidirectional/strided_slice_8/stack_2*
_output_shapes
:@*
Index0*
T0

bidirectional/BiasAdd_1BiasAddbidirectional/MatMul_1bidirectional/strided_slice_8*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
{
bidirectional/ReadVariableOp_6ReadVariableOpbidirectional/forward_lstm/bias*
_output_shapes	
:*
dtype0
n
#bidirectional/strided_slice_9/stackConst*
dtype0*
_output_shapes
:*
valueB:
p
%bidirectional/strided_slice_9/stack_1Const*
_output_shapes
:*
valueB:À*
dtype0
o
%bidirectional/strided_slice_9/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ò
bidirectional/strided_slice_9StridedSlicebidirectional/ReadVariableOp_6#bidirectional/strided_slice_9/stack%bidirectional/strided_slice_9/stack_1%bidirectional/strided_slice_9/stack_2*
_output_shapes
:@*
Index0*
T0

bidirectional/BiasAdd_2BiasAddbidirectional/MatMul_2bidirectional/strided_slice_9*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
{
bidirectional/ReadVariableOp_7ReadVariableOpbidirectional/forward_lstm/bias*
dtype0*
_output_shapes	
:
o
$bidirectional/strided_slice_10/stackConst*
valueB:À*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_10/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_10/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional/strided_slice_10StridedSlicebidirectional/ReadVariableOp_7$bidirectional/strided_slice_10/stack&bidirectional/strided_slice_10/stack_1&bidirectional/strided_slice_10/stack_2*
_output_shapes
:@*
T0*
Index0*
end_mask

bidirectional/BiasAdd_3BiasAddbidirectional/MatMul_3bidirectional/strided_slice_10*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_8ReadVariableOp+bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
:	@*
dtype0
u
$bidirectional/strided_slice_11/stackConst*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_11/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_11/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      

bidirectional/strided_slice_11StridedSlicebidirectional/ReadVariableOp_8$bidirectional/strided_slice_11/stack&bidirectional/strided_slice_11/stack_1&bidirectional/strided_slice_11/stack_2*
_output_shapes

:@@*
T0*
Index0*

begin_mask*
end_mask

bidirectional/MatMul_4MatMulbidirectional/zerosbidirectional/strided_slice_11*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
y
bidirectional/addAddbidirectional/BiasAddbidirectional/MatMul_4*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
X
bidirectional/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍÌL>
r
bidirectional/mulMulbidirectional/mul/xbidirectional/add*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Z
bidirectional/add_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
v
bidirectional/add_1Addbidirectional/mulbidirectional/add_1/y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
X
bidirectional/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Z
bidirectional/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

#bidirectional/clip_by_value/MinimumMinimumbidirectional/add_1bidirectional/Const_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/clip_by_valueMaximum#bidirectional/clip_by_value/Minimumbidirectional/Const*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/ReadVariableOp_9ReadVariableOp+bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_12/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_12/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_12/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      

bidirectional/strided_slice_12StridedSlicebidirectional/ReadVariableOp_9$bidirectional/strided_slice_12/stack&bidirectional/strided_slice_12/stack_1&bidirectional/strided_slice_12/stack_2*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@

bidirectional/MatMul_5MatMulbidirectional/zerosbidirectional/strided_slice_12*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/add_2Addbidirectional/BiasAdd_1bidirectional/MatMul_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Z
bidirectional/mul_1/xConst*
valueB
 *ÍÌL>*
dtype0*
_output_shapes
: 
x
bidirectional/mul_1Mulbidirectional/mul_1/xbidirectional/add_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
Z
bidirectional/add_3/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
x
bidirectional/add_3Addbidirectional/mul_1bidirectional/add_3/y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Z
bidirectional/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_3Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%bidirectional/clip_by_value_1/MinimumMinimumbidirectional/add_3bidirectional/Const_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/clip_by_value_1Maximum%bidirectional/clip_by_value_1/Minimumbidirectional/Const_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/mul_2Mulbidirectional/clip_by_value_1bidirectional/zeros_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_10ReadVariableOp+bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_13/stackConst*
valueB"       *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_13/stack_1Const*
valueB"    À   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_13/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0

bidirectional/strided_slice_13StridedSlicebidirectional/ReadVariableOp_10$bidirectional/strided_slice_13/stack&bidirectional/strided_slice_13/stack_1&bidirectional/strided_slice_13/stack_2*

begin_mask*
T0*
Index0*
end_mask*
_output_shapes

:@@

bidirectional/MatMul_6MatMulbidirectional/zerosbidirectional/strided_slice_13*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/add_4Addbidirectional/BiasAdd_2bidirectional/MatMul_6*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
a
bidirectional/TanhTanhbidirectional/add_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/mul_3Mulbidirectional/clip_by_valuebidirectional/Tanh*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
v
bidirectional/add_5Addbidirectional/mul_2bidirectional/mul_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_11ReadVariableOp+bidirectional/forward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_14/stackConst*
valueB"    À   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_14/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_14/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      

bidirectional/strided_slice_14StridedSlicebidirectional/ReadVariableOp_11$bidirectional/strided_slice_14/stack&bidirectional/strided_slice_14/stack_1&bidirectional/strided_slice_14/stack_2*
_output_shapes

:@@*
T0*
Index0*

begin_mask*
end_mask

bidirectional/MatMul_7MatMulbidirectional/zerosbidirectional/strided_slice_14*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/add_6Addbidirectional/BiasAdd_3bidirectional/MatMul_7*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
Z
bidirectional/mul_4/xConst*
valueB
 *ÍÌL>*
dtype0*
_output_shapes
: 
x
bidirectional/mul_4Mulbidirectional/mul_4/xbidirectional/add_6*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
Z
bidirectional/add_7/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
x
bidirectional/add_7Addbidirectional/mul_4bidirectional/add_7/y*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
Z
bidirectional/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_5Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%bidirectional/clip_by_value_2/MinimumMinimumbidirectional/add_7bidirectional/Const_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/clip_by_value_2Maximum%bidirectional/clip_by_value_2/Minimumbidirectional/Const_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
c
bidirectional/Tanh_1Tanhbidirectional/add_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/mul_5Mulbidirectional/clip_by_value_2bidirectional/Tanh_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
º
bidirectional/TensorArray_1TensorArrayV3bidirectional/strided_slice_1*
identical_element_shapes(*"
tensor_array_nameoutput_ta_0*
dtype0*
_output_shapes

:: 
T
bidirectional/timeConst*
value	B : *
dtype0*
_output_shapes
: 
¥
bidirectional/while/EnterEnterbidirectional/time*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context*
T0
²
bidirectional/while/Enter_1Enterbidirectional/TensorArray_1:1*
T0*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context
¹
bidirectional/while/Enter_2Enterbidirectional/zeros*
T0*
parallel_iterations *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*1

frame_name#!bidirectional/while/while_context
»
bidirectional/while/Enter_3Enterbidirectional/zeros_1*
parallel_iterations *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*1

frame_name#!bidirectional/while/while_context*
T0

bidirectional/while/MergeMergebidirectional/while/Enter!bidirectional/while/NextIteration*
N*
_output_shapes
: : *
T0

bidirectional/while/Merge_1Mergebidirectional/while/Enter_1#bidirectional/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
£
bidirectional/while/Merge_2Mergebidirectional/while/Enter_2#bidirectional/while/NextIteration_2*
T0*
N*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@: 
£
bidirectional/while/Merge_3Mergebidirectional/while/Enter_3#bidirectional/while/NextIteration_3*
T0*
N*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@: 
|
bidirectional/while/LessLessbidirectional/while/Mergebidirectional/while/Less/Enter*
T0*
_output_shapes
: 
È
bidirectional/while/Less/EnterEnterbidirectional/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context
Z
bidirectional/while/LoopCondLoopCondbidirectional/while/Less*
_output_shapes
: 
®
bidirectional/while/SwitchSwitchbidirectional/while/Mergebidirectional/while/LoopCond*
T0*,
_class"
 loc:@bidirectional/while/Merge*
_output_shapes
: : 
´
bidirectional/while/Switch_1Switchbidirectional/while/Merge_1bidirectional/while/LoopCond*
T0*.
_class$
" loc:@bidirectional/while/Merge_1*
_output_shapes
: : 
Ö
bidirectional/while/Switch_2Switchbidirectional/while/Merge_2bidirectional/while/LoopCond*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*
T0*.
_class$
" loc:@bidirectional/while/Merge_2
Ö
bidirectional/while/Switch_3Switchbidirectional/while/Merge_3bidirectional/while/LoopCond*
T0*.
_class$
" loc:@bidirectional/while/Merge_3*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
g
bidirectional/while/IdentityIdentitybidirectional/while/Switch:1*
T0*
_output_shapes
: 
k
bidirectional/while/Identity_1Identitybidirectional/while/Switch_1:1*
_output_shapes
: *
T0
|
bidirectional/while/Identity_2Identitybidirectional/while/Switch_2:1*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
|
bidirectional/while/Identity_3Identitybidirectional/while/Switch_3:1*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
ê
%bidirectional/while/TensorArrayReadV3TensorArrayReadV3+bidirectional/while/TensorArrayReadV3/Enterbidirectional/while/Identity-bidirectional/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Õ
+bidirectional/while/TensorArrayReadV3/EnterEnterbidirectional/TensorArray*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*1

frame_name#!bidirectional/while/while_context

-bidirectional/while/TensorArrayReadV3/Enter_1EnterHbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *1

frame_name#!bidirectional/while/while_context*
T0*
is_constant(*
parallel_iterations 
«
"bidirectional/while/ReadVariableOpReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes
:	@
Ö
(bidirectional/while/ReadVariableOp/EnterEnter!bidirectional/forward_lstm/kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context

'bidirectional/while/strided_slice/stackConst^bidirectional/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

)bidirectional/while/strided_slice/stack_1Const^bidirectional/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:

)bidirectional/while/strided_slice/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
¬
!bidirectional/while/strided_sliceStridedSlice"bidirectional/while/ReadVariableOp'bidirectional/while/strided_slice/stack)bidirectional/while/strided_slice/stack_1)bidirectional/while/strided_slice/stack_2*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@
 
bidirectional/while/MatMulMatMul%bidirectional/while/TensorArrayReadV3!bidirectional/while/strided_slice*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
­
$bidirectional/while/ReadVariableOp_1ReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity*
_output_shapes
:	@*
dtype0

)bidirectional/while/strided_slice_1/stackConst^bidirectional/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_1/stack_1Const^bidirectional/while/Identity*
valueB"       *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_1/stack_2Const^bidirectional/while/Identity*
dtype0*
_output_shapes
:*
valueB"      
¶
#bidirectional/while/strided_slice_1StridedSlice$bidirectional/while/ReadVariableOp_1)bidirectional/while/strided_slice_1/stack+bidirectional/while/strided_slice_1/stack_1+bidirectional/while/strided_slice_1/stack_2*
end_mask*
_output_shapes

:@@*

begin_mask*
T0*
Index0
¤
bidirectional/while/MatMul_1MatMul%bidirectional/while/TensorArrayReadV3#bidirectional/while/strided_slice_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
­
$bidirectional/while/ReadVariableOp_2ReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes
:	@

)bidirectional/while/strided_slice_2/stackConst^bidirectional/while/Identity*
valueB"       *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_2/stack_1Const^bidirectional/while/Identity*
valueB"    À   *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_2/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
¶
#bidirectional/while/strided_slice_2StridedSlice$bidirectional/while/ReadVariableOp_2)bidirectional/while/strided_slice_2/stack+bidirectional/while/strided_slice_2/stack_1+bidirectional/while/strided_slice_2/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:@@
¤
bidirectional/while/MatMul_2MatMul%bidirectional/while/TensorArrayReadV3#bidirectional/while/strided_slice_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
­
$bidirectional/while/ReadVariableOp_3ReadVariableOp(bidirectional/while/ReadVariableOp/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes
:	@

)bidirectional/while/strided_slice_3/stackConst^bidirectional/while/Identity*
valueB"    À   *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_3/stack_1Const^bidirectional/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_3/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
¶
#bidirectional/while/strided_slice_3StridedSlice$bidirectional/while/ReadVariableOp_3)bidirectional/while/strided_slice_3/stack+bidirectional/while/strided_slice_3/stack_1+bidirectional/while/strided_slice_3/stack_2*

begin_mask*
T0*
Index0*
end_mask*
_output_shapes

:@@
¤
bidirectional/while/MatMul_3MatMul%bidirectional/while/TensorArrayReadV3#bidirectional/while/strided_slice_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
«
$bidirectional/while/ReadVariableOp_4ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes	
:
Ö
*bidirectional/while/ReadVariableOp_4/EnterEnterbidirectional/forward_lstm/bias*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context

)bidirectional/while/strided_slice_4/stackConst^bidirectional/while/Identity*
valueB: *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_4/stack_1Const^bidirectional/while/Identity*
valueB:@*
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_4/stack_2Const^bidirectional/while/Identity*
_output_shapes
:*
valueB:*
dtype0
¢
#bidirectional/while/strided_slice_4StridedSlice$bidirectional/while/ReadVariableOp_4)bidirectional/while/strided_slice_4/stack+bidirectional/while/strided_slice_4/stack_1+bidirectional/while/strided_slice_4/stack_2*
Index0*
T0*

begin_mask*
_output_shapes
:@

bidirectional/while/BiasAddBiasAddbidirectional/while/MatMul#bidirectional/while/strided_slice_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
«
$bidirectional/while/ReadVariableOp_5ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes	
:

)bidirectional/while/strided_slice_5/stackConst^bidirectional/while/Identity*
dtype0*
_output_shapes
:*
valueB:@

+bidirectional/while/strided_slice_5/stack_1Const^bidirectional/while/Identity*
_output_shapes
:*
valueB:*
dtype0

+bidirectional/while/strided_slice_5/stack_2Const^bidirectional/while/Identity*
dtype0*
_output_shapes
:*
valueB:

#bidirectional/while/strided_slice_5StridedSlice$bidirectional/while/ReadVariableOp_5)bidirectional/while/strided_slice_5/stack+bidirectional/while/strided_slice_5/stack_1+bidirectional/while/strided_slice_5/stack_2*
Index0*
T0*
_output_shapes
:@

bidirectional/while/BiasAdd_1BiasAddbidirectional/while/MatMul_1#bidirectional/while/strided_slice_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
«
$bidirectional/while/ReadVariableOp_6ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes	
:

)bidirectional/while/strided_slice_6/stackConst^bidirectional/while/Identity*
dtype0*
_output_shapes
:*
valueB:

+bidirectional/while/strided_slice_6/stack_1Const^bidirectional/while/Identity*
valueB:À*
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_6/stack_2Const^bidirectional/while/Identity*
valueB:*
dtype0*
_output_shapes
:

#bidirectional/while/strided_slice_6StridedSlice$bidirectional/while/ReadVariableOp_6)bidirectional/while/strided_slice_6/stack+bidirectional/while/strided_slice_6/stack_1+bidirectional/while/strided_slice_6/stack_2*
Index0*
T0*
_output_shapes
:@

bidirectional/while/BiasAdd_2BiasAddbidirectional/while/MatMul_2#bidirectional/while/strided_slice_6*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
«
$bidirectional/while/ReadVariableOp_7ReadVariableOp*bidirectional/while/ReadVariableOp_4/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes	
:

)bidirectional/while/strided_slice_7/stackConst^bidirectional/while/Identity*
dtype0*
_output_shapes
:*
valueB:À

+bidirectional/while/strided_slice_7/stack_1Const^bidirectional/while/Identity*
valueB: *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_7/stack_2Const^bidirectional/while/Identity*
valueB:*
dtype0*
_output_shapes
:
 
#bidirectional/while/strided_slice_7StridedSlice$bidirectional/while/ReadVariableOp_7)bidirectional/while/strided_slice_7/stack+bidirectional/while/strided_slice_7/stack_1+bidirectional/while/strided_slice_7/stack_2*
Index0*
T0*
end_mask*
_output_shapes
:@

bidirectional/while/BiasAdd_3BiasAddbidirectional/while/MatMul_3#bidirectional/while/strided_slice_7*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
¯
$bidirectional/while/ReadVariableOp_8ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes
:	@
â
*bidirectional/while/ReadVariableOp_8/EnterEnter+bidirectional/forward_lstm/recurrent_kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *1

frame_name#!bidirectional/while/while_context

)bidirectional/while/strided_slice_8/stackConst^bidirectional/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_8/stack_1Const^bidirectional/while/Identity*
valueB"    @   *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_8/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
¶
#bidirectional/while/strided_slice_8StridedSlice$bidirectional/while/ReadVariableOp_8)bidirectional/while/strided_slice_8/stack+bidirectional/while/strided_slice_8/stack_1+bidirectional/while/strided_slice_8/stack_2*

begin_mask*
T0*
Index0*
end_mask*
_output_shapes

:@@

bidirectional/while/MatMul_4MatMulbidirectional/while/Identity_2#bidirectional/while/strided_slice_8*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/addAddbidirectional/while/BiasAddbidirectional/while/MatMul_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/while/mul/xConst^bidirectional/while/Identity*
valueB
 *ÍÌL>*
dtype0*
_output_shapes
: 

bidirectional/while/mulMulbidirectional/while/mul/xbidirectional/while/add*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while/add_1/yConst^bidirectional/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while/add_1Addbidirectional/while/mulbidirectional/while/add_1/y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/while/ConstConst^bidirectional/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *    

bidirectional/while/Const_1Const^bidirectional/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

)bidirectional/while/clip_by_value/MinimumMinimumbidirectional/while/add_1bidirectional/while/Const_1*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
¤
!bidirectional/while/clip_by_valueMaximum)bidirectional/while/clip_by_value/Minimumbidirectional/while/Const*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
¯
$bidirectional/while/ReadVariableOp_9ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes
:	@

)bidirectional/while/strided_slice_9/stackConst^bidirectional/while/Identity*
dtype0*
_output_shapes
:*
valueB"    @   

+bidirectional/while/strided_slice_9/stack_1Const^bidirectional/while/Identity*
valueB"       *
dtype0*
_output_shapes
:

+bidirectional/while/strided_slice_9/stack_2Const^bidirectional/while/Identity*
dtype0*
_output_shapes
:*
valueB"      
¶
#bidirectional/while/strided_slice_9StridedSlice$bidirectional/while/ReadVariableOp_9)bidirectional/while/strided_slice_9/stack+bidirectional/while/strided_slice_9/stack_1+bidirectional/while/strided_slice_9/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:@@

bidirectional/while/MatMul_5MatMulbidirectional/while/Identity_2#bidirectional/while/strided_slice_9*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/add_2Addbidirectional/while/BiasAdd_1bidirectional/while/MatMul_5*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while/mul_1/xConst^bidirectional/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *ÍÌL>

bidirectional/while/mul_1Mulbidirectional/while/mul_1/xbidirectional/while/add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/add_3/yConst^bidirectional/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while/add_3Addbidirectional/while/mul_1bidirectional/while/add_3/y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/Const_2Const^bidirectional/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while/Const_3Const^bidirectional/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
 
+bidirectional/while/clip_by_value_1/MinimumMinimumbidirectional/while/add_3bidirectional/while/Const_3*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
ª
#bidirectional/while/clip_by_value_1Maximum+bidirectional/while/clip_by_value_1/Minimumbidirectional/while/Const_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while/mul_2Mul#bidirectional/while/clip_by_value_1bidirectional/while/Identity_3*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
°
%bidirectional/while/ReadVariableOp_10ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity*
_output_shapes
:	@*
dtype0

*bidirectional/while/strided_slice_10/stackConst^bidirectional/while/Identity*
valueB"       *
dtype0*
_output_shapes
:

,bidirectional/while/strided_slice_10/stack_1Const^bidirectional/while/Identity*
valueB"    À   *
dtype0*
_output_shapes
:

,bidirectional/while/strided_slice_10/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
»
$bidirectional/while/strided_slice_10StridedSlice%bidirectional/while/ReadVariableOp_10*bidirectional/while/strided_slice_10/stack,bidirectional/while/strided_slice_10/stack_1,bidirectional/while/strided_slice_10/stack_2*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@

bidirectional/while/MatMul_6MatMulbidirectional/while/Identity_2$bidirectional/while/strided_slice_10*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/add_4Addbidirectional/while/BiasAdd_2bidirectional/while/MatMul_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
m
bidirectional/while/TanhTanhbidirectional/while/add_4*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while/mul_3Mul!bidirectional/while/clip_by_valuebidirectional/while/Tanh*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while/add_5Addbidirectional/while/mul_2bidirectional/while/mul_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
°
%bidirectional/while/ReadVariableOp_11ReadVariableOp*bidirectional/while/ReadVariableOp_8/Enter^bidirectional/while/Identity*
dtype0*
_output_shapes
:	@

*bidirectional/while/strided_slice_11/stackConst^bidirectional/while/Identity*
valueB"    À   *
dtype0*
_output_shapes
:

,bidirectional/while/strided_slice_11/stack_1Const^bidirectional/while/Identity*
valueB"        *
dtype0*
_output_shapes
:

,bidirectional/while/strided_slice_11/stack_2Const^bidirectional/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
»
$bidirectional/while/strided_slice_11StridedSlice%bidirectional/while/ReadVariableOp_11*bidirectional/while/strided_slice_11/stack,bidirectional/while/strided_slice_11/stack_1,bidirectional/while/strided_slice_11/stack_2*
end_mask*
_output_shapes

:@@*

begin_mask*
T0*
Index0

bidirectional/while/MatMul_7MatMulbidirectional/while/Identity_2$bidirectional/while/strided_slice_11*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/add_6Addbidirectional/while/BiasAdd_3bidirectional/while/MatMul_7*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/mul_4/xConst^bidirectional/while/Identity*
_output_shapes
: *
valueB
 *ÍÌL>*
dtype0

bidirectional/while/mul_4Mulbidirectional/while/mul_4/xbidirectional/while/add_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/add_7/yConst^bidirectional/while/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while/add_7Addbidirectional/while/mul_4bidirectional/while/add_7/y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/Const_4Const^bidirectional/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while/Const_5Const^bidirectional/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
 
+bidirectional/while/clip_by_value_2/MinimumMinimumbidirectional/while/add_7bidirectional/while/Const_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
ª
#bidirectional/while/clip_by_value_2Maximum+bidirectional/while/clip_by_value_2/Minimumbidirectional/while/Const_4*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
o
bidirectional/while/Tanh_1Tanhbidirectional/while/add_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while/mul_5Mul#bidirectional/while/clip_by_value_2bidirectional/while/Tanh_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
´
7bidirectional/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/Enterbidirectional/while/Identitybidirectional/while/mul_5bidirectional/while/Identity_1*
T0*,
_class"
 loc:@bidirectional/while/mul_5*
_output_shapes
: 

=bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterbidirectional/TensorArray_1*
is_constant(*
_output_shapes
:*1

frame_name#!bidirectional/while/while_context*
T0*,
_class"
 loc:@bidirectional/while/mul_5*
parallel_iterations 
|
bidirectional/while/add_8/yConst^bidirectional/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
|
bidirectional/while/add_8Addbidirectional/while/Identitybidirectional/while/add_8/y*
T0*
_output_shapes
: 
n
!bidirectional/while/NextIterationNextIterationbidirectional/while/add_8*
T0*
_output_shapes
: 

#bidirectional/while/NextIteration_1NextIteration7bidirectional/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

#bidirectional/while/NextIteration_2NextIterationbidirectional/while/mul_5*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

#bidirectional/while/NextIteration_3NextIterationbidirectional/while/add_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
]
bidirectional/while/ExitExitbidirectional/while/Switch*
_output_shapes
: *
T0
a
bidirectional/while/Exit_1Exitbidirectional/while/Switch_1*
_output_shapes
: *
T0
r
bidirectional/while/Exit_2Exitbidirectional/while/Switch_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
r
bidirectional/while/Exit_3Exitbidirectional/while/Switch_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Æ
0bidirectional/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3bidirectional/TensorArray_1bidirectional/while/Exit_1*.
_class$
" loc:@bidirectional/TensorArray_1*
_output_shapes
: 

*bidirectional/TensorArrayStack/range/startConst*
value	B : *.
_class$
" loc:@bidirectional/TensorArray_1*
dtype0*
_output_shapes
: 

*bidirectional/TensorArrayStack/range/deltaConst*
value	B :*.
_class$
" loc:@bidirectional/TensorArray_1*
dtype0*
_output_shapes
: 

$bidirectional/TensorArrayStack/rangeRange*bidirectional/TensorArrayStack/range/start0bidirectional/TensorArrayStack/TensorArraySizeV3*bidirectional/TensorArrayStack/range/delta*.
_class$
" loc:@bidirectional/TensorArray_1*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á
2bidirectional/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3bidirectional/TensorArray_1$bidirectional/TensorArrayStack/rangebidirectional/while/Exit_1*
dtype0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
element_shape:ÿÿÿÿÿÿÿÿÿ@*.
_class$
" loc:@bidirectional/TensorArray_1
w
$bidirectional/strided_slice_15/stackConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_15/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_15/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¯
bidirectional/strided_slice_15StridedSlice2bidirectional/TensorArrayStack/TensorArrayGatherV3$bidirectional/strided_slice_15/stack&bidirectional/strided_slice_15/stack_1&bidirectional/strided_slice_15/stack_2*
T0*
Index0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask
s
bidirectional/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
¹
bidirectional/transpose_1	Transpose2bidirectional/TensorArrayStack/TensorArrayGatherV3bidirectional/transpose_1/perm*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0
j
bidirectional/Shape_2Shape%embedding/embedding_lookup/Identity_1*
_output_shapes
:*
T0
n
$bidirectional/strided_slice_16/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_16/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
p
&bidirectional/strided_slice_16/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

bidirectional/strided_slice_16StridedSlicebidirectional/Shape_2$bidirectional/strided_slice_16/stack&bidirectional/strided_slice_16/stack_1&bidirectional/strided_slice_16/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
]
bidirectional/zeros_2/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: 
~
bidirectional/zeros_2/mulMulbidirectional/strided_slice_16bidirectional/zeros_2/mul/y*
T0*
_output_shapes
: 
_
bidirectional/zeros_2/Less/yConst*
value
B :è*
dtype0*
_output_shapes
: 
|
bidirectional/zeros_2/LessLessbidirectional/zeros_2/mulbidirectional/zeros_2/Less/y*
_output_shapes
: *
T0
`
bidirectional/zeros_2/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@

bidirectional/zeros_2/packedPackbidirectional/strided_slice_16bidirectional/zeros_2/packed/1*
T0*
N*
_output_shapes
:
`
bidirectional/zeros_2/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

bidirectional/zeros_2Fillbidirectional/zeros_2/packedbidirectional/zeros_2/Const*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
]
bidirectional/zeros_3/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: 
~
bidirectional/zeros_3/mulMulbidirectional/strided_slice_16bidirectional/zeros_3/mul/y*
_output_shapes
: *
T0
_
bidirectional/zeros_3/Less/yConst*
value
B :è*
dtype0*
_output_shapes
: 
|
bidirectional/zeros_3/LessLessbidirectional/zeros_3/mulbidirectional/zeros_3/Less/y*
T0*
_output_shapes
: 
`
bidirectional/zeros_3/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@

bidirectional/zeros_3/packedPackbidirectional/strided_slice_16bidirectional/zeros_3/packed/1*
N*
_output_shapes
:*
T0
`
bidirectional/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/zeros_3Fillbidirectional/zeros_3/packedbidirectional/zeros_3/Const*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
s
bidirectional/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
¬
bidirectional/transpose_2	Transpose%embedding/embedding_lookup/Identity_1bidirectional/transpose_2/perm*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
^
bidirectional/Shape_3Shapebidirectional/transpose_2*
_output_shapes
:*
T0
n
$bidirectional/strided_slice_17/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_17/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_17/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

bidirectional/strided_slice_17StridedSlicebidirectional/Shape_3$bidirectional/strided_slice_17/stack&bidirectional/strided_slice_17/stack_1&bidirectional/strided_slice_17/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
º
bidirectional/TensorArray_2TensorArrayV3bidirectional/strided_slice_17*
identical_element_shapes(*!
tensor_array_name
input_ta_0*
dtype0*
_output_shapes

:: 
f
bidirectional/ReverseV2/axisConst*
valueB: *
dtype0*
_output_shapes
:

bidirectional/ReverseV2	ReverseV2bidirectional/transpose_2bidirectional/ReverseV2/axis*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
T0
o
(bidirectional/TensorArrayUnstack_1/ShapeShapebidirectional/ReverseV2*
_output_shapes
:*
T0

6bidirectional/TensorArrayUnstack_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

8bidirectional/TensorArrayUnstack_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

8bidirectional/TensorArrayUnstack_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ü
0bidirectional/TensorArrayUnstack_1/strided_sliceStridedSlice(bidirectional/TensorArrayUnstack_1/Shape6bidirectional/TensorArrayUnstack_1/strided_slice/stack8bidirectional/TensorArrayUnstack_1/strided_slice/stack_18bidirectional/TensorArrayUnstack_1/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
p
.bidirectional/TensorArrayUnstack_1/range/startConst*
_output_shapes
: *
value	B : *
dtype0
p
.bidirectional/TensorArrayUnstack_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
è
(bidirectional/TensorArrayUnstack_1/rangeRange.bidirectional/TensorArrayUnstack_1/range/start0bidirectional/TensorArrayUnstack_1/strided_slice.bidirectional/TensorArrayUnstack_1/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
®
Jbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3bidirectional/TensorArray_2(bidirectional/TensorArrayUnstack_1/rangebidirectional/ReverseV2bidirectional/TensorArray_2:1*
_output_shapes
: *
T0**
_class 
loc:@bidirectional/ReverseV2
n
$bidirectional/strided_slice_18/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_18/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
p
&bidirectional/strided_slice_18/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional/strided_slice_18StridedSlicebidirectional/transpose_2$bidirectional/strided_slice_18/stack&bidirectional/strided_slice_18/stack_1&bidirectional/strided_slice_18/stack_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask*
T0*
Index0

bidirectional/ReadVariableOp_12ReadVariableOp"bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_19/stackConst*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_19/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_19/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_19StridedSlicebidirectional/ReadVariableOp_12$bidirectional/strided_slice_19/stack&bidirectional/strided_slice_19/stack_1&bidirectional/strided_slice_19/stack_2*
end_mask*
_output_shapes

:@@*

begin_mask*
Index0*
T0

bidirectional/MatMul_8MatMulbidirectional/strided_slice_18bidirectional/strided_slice_19*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_13ReadVariableOp"bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_20/stackConst*
dtype0*
_output_shapes
:*
valueB"    @   
w
&bidirectional/strided_slice_20/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
w
&bidirectional/strided_slice_20/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_20StridedSlicebidirectional/ReadVariableOp_13$bidirectional/strided_slice_20/stack&bidirectional/strided_slice_20/stack_1&bidirectional/strided_slice_20/stack_2*
end_mask*
_output_shapes

:@@*
Index0*
T0*

begin_mask

bidirectional/MatMul_9MatMulbidirectional/strided_slice_18bidirectional/strided_slice_20*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_14ReadVariableOp"bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_21/stackConst*
valueB"       *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_21/stack_1Const*
valueB"    À   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_21/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      

bidirectional/strided_slice_21StridedSlicebidirectional/ReadVariableOp_14$bidirectional/strided_slice_21/stack&bidirectional/strided_slice_21/stack_1&bidirectional/strided_slice_21/stack_2*
end_mask*
_output_shapes

:@@*

begin_mask*
T0*
Index0

bidirectional/MatMul_10MatMulbidirectional/strided_slice_18bidirectional/strided_slice_21*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_15ReadVariableOp"bidirectional/backward_lstm/kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_22/stackConst*
valueB"    À   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_22/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_22/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_22StridedSlicebidirectional/ReadVariableOp_15$bidirectional/strided_slice_22/stack&bidirectional/strided_slice_22/stack_1&bidirectional/strided_slice_22/stack_2*
_output_shapes

:@@*

begin_mask*
T0*
Index0*
end_mask

bidirectional/MatMul_11MatMulbidirectional/strided_slice_18bidirectional/strided_slice_22*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/ReadVariableOp_16ReadVariableOp bidirectional/backward_lstm/bias*
dtype0*
_output_shapes	
:
n
$bidirectional/strided_slice_23/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_23/stack_1Const*
valueB:@*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_23/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional/strided_slice_23StridedSlicebidirectional/ReadVariableOp_16$bidirectional/strided_slice_23/stack&bidirectional/strided_slice_23/stack_1&bidirectional/strided_slice_23/stack_2*
_output_shapes
:@*

begin_mask*
Index0*
T0

bidirectional/BiasAdd_4BiasAddbidirectional/MatMul_8bidirectional/strided_slice_23*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/ReadVariableOp_17ReadVariableOp bidirectional/backward_lstm/bias*
dtype0*
_output_shapes	
:
n
$bidirectional/strided_slice_24/stackConst*
valueB:@*
dtype0*
_output_shapes
:
q
&bidirectional/strided_slice_24/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
p
&bidirectional/strided_slice_24/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
÷
bidirectional/strided_slice_24StridedSlicebidirectional/ReadVariableOp_17$bidirectional/strided_slice_24/stack&bidirectional/strided_slice_24/stack_1&bidirectional/strided_slice_24/stack_2*
_output_shapes
:@*
T0*
Index0

bidirectional/BiasAdd_5BiasAddbidirectional/MatMul_9bidirectional/strided_slice_24*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/ReadVariableOp_18ReadVariableOp bidirectional/backward_lstm/bias*
dtype0*
_output_shapes	
:
o
$bidirectional/strided_slice_25/stackConst*
_output_shapes
:*
valueB:*
dtype0
q
&bidirectional/strided_slice_25/stack_1Const*
valueB:À*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_25/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
÷
bidirectional/strided_slice_25StridedSlicebidirectional/ReadVariableOp_18$bidirectional/strided_slice_25/stack&bidirectional/strided_slice_25/stack_1&bidirectional/strided_slice_25/stack_2*
_output_shapes
:@*
T0*
Index0

bidirectional/BiasAdd_6BiasAddbidirectional/MatMul_10bidirectional/strided_slice_25*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
}
bidirectional/ReadVariableOp_19ReadVariableOp bidirectional/backward_lstm/bias*
dtype0*
_output_shapes	
:
o
$bidirectional/strided_slice_26/stackConst*
valueB:À*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_26/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_26/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional/strided_slice_26StridedSlicebidirectional/ReadVariableOp_19$bidirectional/strided_slice_26/stack&bidirectional/strided_slice_26/stack_1&bidirectional/strided_slice_26/stack_2*
T0*
Index0*
end_mask*
_output_shapes
:@

bidirectional/BiasAdd_7BiasAddbidirectional/MatMul_11bidirectional/strided_slice_26*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_20ReadVariableOp,bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_27/stackConst*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_27/stack_1Const*
valueB"    @   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_27/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_27StridedSlicebidirectional/ReadVariableOp_20$bidirectional/strided_slice_27/stack&bidirectional/strided_slice_27/stack_1&bidirectional/strided_slice_27/stack_2*
_output_shapes

:@@*

begin_mask*
T0*
Index0*
end_mask

bidirectional/MatMul_12MatMulbidirectional/zeros_2bidirectional/strided_slice_27*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
~
bidirectional/add_8Addbidirectional/BiasAdd_4bidirectional/MatMul_12*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Z
bidirectional/mul_6/xConst*
valueB
 *ÍÌL>*
dtype0*
_output_shapes
: 
x
bidirectional/mul_6Mulbidirectional/mul_6/xbidirectional/add_8*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Z
bidirectional/add_9/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
x
bidirectional/add_9Addbidirectional/mul_6bidirectional/add_9/y*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
Z
bidirectional/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_7Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%bidirectional/clip_by_value_3/MinimumMinimumbidirectional/add_9bidirectional/Const_7*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/clip_by_value_3Maximum%bidirectional/clip_by_value_3/Minimumbidirectional/Const_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_21ReadVariableOp,bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
:	@*
dtype0
u
$bidirectional/strided_slice_28/stackConst*
valueB"    @   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_28/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_28/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_28StridedSlicebidirectional/ReadVariableOp_21$bidirectional/strided_slice_28/stack&bidirectional/strided_slice_28/stack_1&bidirectional/strided_slice_28/stack_2*
end_mask*
_output_shapes

:@@*

begin_mask*
T0*
Index0

bidirectional/MatMul_13MatMulbidirectional/zeros_2bidirectional/strided_slice_28*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/add_10Addbidirectional/BiasAdd_5bidirectional/MatMul_13*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Z
bidirectional/mul_7/xConst*
dtype0*
_output_shapes
: *
valueB
 *ÍÌL>
y
bidirectional/mul_7Mulbidirectional/mul_7/xbidirectional/add_10*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
[
bidirectional/add_11/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
z
bidirectional/add_11Addbidirectional/mul_7bidirectional/add_11/y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Z
bidirectional/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
bidirectional/Const_9Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

%bidirectional/clip_by_value_4/MinimumMinimumbidirectional/add_11bidirectional/Const_9*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/clip_by_value_4Maximum%bidirectional/clip_by_value_4/Minimumbidirectional/Const_8*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/mul_8Mulbidirectional/clip_by_value_4bidirectional/zeros_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_22ReadVariableOp,bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_29/stackConst*
valueB"       *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_29/stack_1Const*
valueB"    À   *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_29/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      

bidirectional/strided_slice_29StridedSlicebidirectional/ReadVariableOp_22$bidirectional/strided_slice_29/stack&bidirectional/strided_slice_29/stack_1&bidirectional/strided_slice_29/stack_2*
end_mask*
_output_shapes

:@@*

begin_mask*
T0*
Index0

bidirectional/MatMul_14MatMulbidirectional/zeros_2bidirectional/strided_slice_29*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/add_12Addbidirectional/BiasAdd_6bidirectional/MatMul_14*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
d
bidirectional/Tanh_2Tanhbidirectional/add_12*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/mul_9Mulbidirectional/clip_by_value_3bidirectional/Tanh_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
w
bidirectional/add_13Addbidirectional/mul_8bidirectional/mul_9*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/ReadVariableOp_23ReadVariableOp,bidirectional/backward_lstm/recurrent_kernel*
dtype0*
_output_shapes
:	@
u
$bidirectional/strided_slice_30/stackConst*
dtype0*
_output_shapes
:*
valueB"    À   
w
&bidirectional/strided_slice_30/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
w
&bidirectional/strided_slice_30/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

bidirectional/strided_slice_30StridedSlicebidirectional/ReadVariableOp_23$bidirectional/strided_slice_30/stack&bidirectional/strided_slice_30/stack_1&bidirectional/strided_slice_30/stack_2*

begin_mask*
T0*
Index0*
end_mask*
_output_shapes

:@@

bidirectional/MatMul_15MatMulbidirectional/zeros_2bidirectional/strided_slice_30*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/add_14Addbidirectional/BiasAdd_7bidirectional/MatMul_15*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
[
bidirectional/mul_10/xConst*
valueB
 *ÍÌL>*
dtype0*
_output_shapes
: 
{
bidirectional/mul_10Mulbidirectional/mul_10/xbidirectional/add_14*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
[
bidirectional/add_15/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
{
bidirectional/add_15Addbidirectional/mul_10bidirectional/add_15/y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
[
bidirectional/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
bidirectional/Const_11Const*
_output_shapes
: *
valueB
 *  ?*
dtype0

%bidirectional/clip_by_value_5/MinimumMinimumbidirectional/add_15bidirectional/Const_11*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/clip_by_value_5Maximum%bidirectional/clip_by_value_5/Minimumbidirectional/Const_10*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
d
bidirectional/Tanh_3Tanhbidirectional/add_13*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/mul_11Mulbidirectional/clip_by_value_5bidirectional/Tanh_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
»
bidirectional/TensorArray_3TensorArrayV3bidirectional/strided_slice_17*
identical_element_shapes(*"
tensor_array_nameoutput_ta_0*
dtype0*
_output_shapes

:: 
V
bidirectional/time_1Const*
value	B : *
dtype0*
_output_shapes
: 
«
bidirectional/while_1/EnterEnterbidirectional/time_1*
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context*
T0*
parallel_iterations 
¶
bidirectional/while_1/Enter_1Enterbidirectional/TensorArray_3:1*
T0*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context
¿
bidirectional/while_1/Enter_2Enterbidirectional/zeros_2*
T0*
parallel_iterations *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*3

frame_name%#bidirectional/while_1/while_context
¿
bidirectional/while_1/Enter_3Enterbidirectional/zeros_3*
parallel_iterations *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*3

frame_name%#bidirectional/while_1/while_context*
T0

bidirectional/while_1/MergeMergebidirectional/while_1/Enter#bidirectional/while_1/NextIteration*
T0*
N*
_output_shapes
: : 

bidirectional/while_1/Merge_1Mergebidirectional/while_1/Enter_1%bidirectional/while_1/NextIteration_1*
N*
_output_shapes
: : *
T0
©
bidirectional/while_1/Merge_2Mergebidirectional/while_1/Enter_2%bidirectional/while_1/NextIteration_2*
T0*
N*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@: 
©
bidirectional/while_1/Merge_3Mergebidirectional/while_1/Enter_3%bidirectional/while_1/NextIteration_3*
N*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@: *
T0

bidirectional/while_1/LessLessbidirectional/while_1/Merge bidirectional/while_1/Less/Enter*
T0*
_output_shapes
: 
Í
 bidirectional/while_1/Less/EnterEnterbidirectional/strided_slice_17*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context
^
bidirectional/while_1/LoopCondLoopCondbidirectional/while_1/Less*
_output_shapes
: 
¶
bidirectional/while_1/SwitchSwitchbidirectional/while_1/Mergebidirectional/while_1/LoopCond*
_output_shapes
: : *
T0*.
_class$
" loc:@bidirectional/while_1/Merge
¼
bidirectional/while_1/Switch_1Switchbidirectional/while_1/Merge_1bidirectional/while_1/LoopCond*
T0*0
_class&
$"loc:@bidirectional/while_1/Merge_1*
_output_shapes
: : 
Þ
bidirectional/while_1/Switch_2Switchbidirectional/while_1/Merge_2bidirectional/while_1/LoopCond*
T0*0
_class&
$"loc:@bidirectional/while_1/Merge_2*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
Þ
bidirectional/while_1/Switch_3Switchbidirectional/while_1/Merge_3bidirectional/while_1/LoopCond*
T0*0
_class&
$"loc:@bidirectional/while_1/Merge_3*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
k
bidirectional/while_1/IdentityIdentitybidirectional/while_1/Switch:1*
T0*
_output_shapes
: 
o
 bidirectional/while_1/Identity_1Identity bidirectional/while_1/Switch_1:1*
_output_shapes
: *
T0

 bidirectional/while_1/Identity_2Identity bidirectional/while_1/Switch_2:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

 bidirectional/while_1/Identity_3Identity bidirectional/while_1/Switch_3:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
ò
'bidirectional/while_1/TensorArrayReadV3TensorArrayReadV3-bidirectional/while_1/TensorArrayReadV3/Enterbidirectional/while_1/Identity/bidirectional/while_1/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Û
-bidirectional/while_1/TensorArrayReadV3/EnterEnterbidirectional/TensorArray_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*3

frame_name%#bidirectional/while_1/while_context

/bidirectional/while_1/TensorArrayReadV3/Enter_1EnterJbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context
±
$bidirectional/while_1/ReadVariableOpReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@
Û
*bidirectional/while_1/ReadVariableOp/EnterEnter"bidirectional/backward_lstm/kernel*
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context*
T0*
is_constant(*
parallel_iterations 

)bidirectional/while_1/strided_slice/stackConst^bidirectional/while_1/Identity*
valueB"        *
dtype0*
_output_shapes
:

+bidirectional/while_1/strided_slice/stack_1Const^bidirectional/while_1/Identity*
valueB"    @   *
dtype0*
_output_shapes
:

+bidirectional/while_1/strided_slice/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:
¶
#bidirectional/while_1/strided_sliceStridedSlice$bidirectional/while_1/ReadVariableOp)bidirectional/while_1/strided_slice/stack+bidirectional/while_1/strided_slice/stack_1+bidirectional/while_1/strided_slice/stack_2*
end_mask*
_output_shapes

:@@*
Index0*
T0*

begin_mask
¦
bidirectional/while_1/MatMulMatMul'bidirectional/while_1/TensorArrayReadV3#bidirectional/while_1/strided_slice*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
³
&bidirectional/while_1/ReadVariableOp_1ReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@

+bidirectional/while_1/strided_slice_1/stackConst^bidirectional/while_1/Identity*
_output_shapes
:*
valueB"    @   *
dtype0

-bidirectional/while_1/strided_slice_1/stack_1Const^bidirectional/while_1/Identity*
valueB"       *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_1/stack_2Const^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:*
valueB"      
À
%bidirectional/while_1/strided_slice_1StridedSlice&bidirectional/while_1/ReadVariableOp_1+bidirectional/while_1/strided_slice_1/stack-bidirectional/while_1/strided_slice_1/stack_1-bidirectional/while_1/strided_slice_1/stack_2*
end_mask*
_output_shapes

:@@*
T0*
Index0*

begin_mask
ª
bidirectional/while_1/MatMul_1MatMul'bidirectional/while_1/TensorArrayReadV3%bidirectional/while_1/strided_slice_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
³
&bidirectional/while_1/ReadVariableOp_2ReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@

+bidirectional/while_1/strided_slice_2/stackConst^bidirectional/while_1/Identity*
valueB"       *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_2/stack_1Const^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:*
valueB"    À   

-bidirectional/while_1/strided_slice_2/stack_2Const^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:*
valueB"      
À
%bidirectional/while_1/strided_slice_2StridedSlice&bidirectional/while_1/ReadVariableOp_2+bidirectional/while_1/strided_slice_2/stack-bidirectional/while_1/strided_slice_2/stack_1-bidirectional/while_1/strided_slice_2/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:@@
ª
bidirectional/while_1/MatMul_2MatMul'bidirectional/while_1/TensorArrayReadV3%bidirectional/while_1/strided_slice_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
³
&bidirectional/while_1/ReadVariableOp_3ReadVariableOp*bidirectional/while_1/ReadVariableOp/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@

+bidirectional/while_1/strided_slice_3/stackConst^bidirectional/while_1/Identity*
valueB"    À   *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_3/stack_1Const^bidirectional/while_1/Identity*
valueB"        *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_3/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:
À
%bidirectional/while_1/strided_slice_3StridedSlice&bidirectional/while_1/ReadVariableOp_3+bidirectional/while_1/strided_slice_3/stack-bidirectional/while_1/strided_slice_3/stack_1-bidirectional/while_1/strided_slice_3/stack_2*
Index0*
T0*

begin_mask*
end_mask*
_output_shapes

:@@
ª
bidirectional/while_1/MatMul_3MatMul'bidirectional/while_1/TensorArrayReadV3%bidirectional/while_1/strided_slice_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
±
&bidirectional/while_1/ReadVariableOp_4ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
_output_shapes	
:*
dtype0
Û
,bidirectional/while_1/ReadVariableOp_4/EnterEnter bidirectional/backward_lstm/bias*
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context*
T0*
is_constant(*
parallel_iterations 

+bidirectional/while_1/strided_slice_4/stackConst^bidirectional/while_1/Identity*
valueB: *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_4/stack_1Const^bidirectional/while_1/Identity*
valueB:@*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_4/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
¬
%bidirectional/while_1/strided_slice_4StridedSlice&bidirectional/while_1/ReadVariableOp_4+bidirectional/while_1/strided_slice_4/stack-bidirectional/while_1/strided_slice_4/stack_1-bidirectional/while_1/strided_slice_4/stack_2*
_output_shapes
:@*

begin_mask*
T0*
Index0

bidirectional/while_1/BiasAddBiasAddbidirectional/while_1/MatMul%bidirectional/while_1/strided_slice_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
±
&bidirectional/while_1/ReadVariableOp_5ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes	
:

+bidirectional/while_1/strided_slice_5/stackConst^bidirectional/while_1/Identity*
valueB:@*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_5/stack_1Const^bidirectional/while_1/Identity*
_output_shapes
:*
valueB:*
dtype0

-bidirectional/while_1/strided_slice_5/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:

%bidirectional/while_1/strided_slice_5StridedSlice&bidirectional/while_1/ReadVariableOp_5+bidirectional/while_1/strided_slice_5/stack-bidirectional/while_1/strided_slice_5/stack_1-bidirectional/while_1/strided_slice_5/stack_2*
Index0*
T0*
_output_shapes
:@
£
bidirectional/while_1/BiasAdd_1BiasAddbidirectional/while_1/MatMul_1%bidirectional/while_1/strided_slice_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
±
&bidirectional/while_1/ReadVariableOp_6ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes	
:

+bidirectional/while_1/strided_slice_6/stackConst^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:

-bidirectional/while_1/strided_slice_6/stack_1Const^bidirectional/while_1/Identity*
valueB:À*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_6/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:

%bidirectional/while_1/strided_slice_6StridedSlice&bidirectional/while_1/ReadVariableOp_6+bidirectional/while_1/strided_slice_6/stack-bidirectional/while_1/strided_slice_6/stack_1-bidirectional/while_1/strided_slice_6/stack_2*
_output_shapes
:@*
T0*
Index0
£
bidirectional/while_1/BiasAdd_2BiasAddbidirectional/while_1/MatMul_2%bidirectional/while_1/strided_slice_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
±
&bidirectional/while_1/ReadVariableOp_7ReadVariableOp,bidirectional/while_1/ReadVariableOp_4/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes	
:

+bidirectional/while_1/strided_slice_7/stackConst^bidirectional/while_1/Identity*
valueB:À*
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_7/stack_1Const^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:*
valueB: 

-bidirectional/while_1/strided_slice_7/stack_2Const^bidirectional/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
ª
%bidirectional/while_1/strided_slice_7StridedSlice&bidirectional/while_1/ReadVariableOp_7+bidirectional/while_1/strided_slice_7/stack-bidirectional/while_1/strided_slice_7/stack_1-bidirectional/while_1/strided_slice_7/stack_2*
end_mask*
_output_shapes
:@*
Index0*
T0
£
bidirectional/while_1/BiasAdd_3BiasAddbidirectional/while_1/MatMul_3%bidirectional/while_1/strided_slice_7*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
µ
&bidirectional/while_1/ReadVariableOp_8ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@
ç
,bidirectional/while_1/ReadVariableOp_8/EnterEnter,bidirectional/backward_lstm/recurrent_kernel*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *3

frame_name%#bidirectional/while_1/while_context

+bidirectional/while_1/strided_slice_8/stackConst^bidirectional/while_1/Identity*
_output_shapes
:*
valueB"        *
dtype0

-bidirectional/while_1/strided_slice_8/stack_1Const^bidirectional/while_1/Identity*
_output_shapes
:*
valueB"    @   *
dtype0

-bidirectional/while_1/strided_slice_8/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:
À
%bidirectional/while_1/strided_slice_8StridedSlice&bidirectional/while_1/ReadVariableOp_8+bidirectional/while_1/strided_slice_8/stack-bidirectional/while_1/strided_slice_8/stack_1-bidirectional/while_1/strided_slice_8/stack_2*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@
£
bidirectional/while_1/MatMul_4MatMul bidirectional/while_1/Identity_2%bidirectional/while_1/strided_slice_8*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/addAddbidirectional/while_1/BiasAddbidirectional/while_1/MatMul_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/mul/xConst^bidirectional/while_1/Identity*
valueB
 *ÍÌL>*
dtype0*
_output_shapes
: 

bidirectional/while_1/mulMulbidirectional/while_1/mul/xbidirectional/while_1/add*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/add_1/yConst^bidirectional/while_1/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while_1/add_1Addbidirectional/while_1/mulbidirectional/while_1/add_1/y*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/ConstConst^bidirectional/while_1/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while_1/Const_1Const^bidirectional/while_1/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
¤
+bidirectional/while_1/clip_by_value/MinimumMinimumbidirectional/while_1/add_1bidirectional/while_1/Const_1*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
ª
#bidirectional/while_1/clip_by_valueMaximum+bidirectional/while_1/clip_by_value/Minimumbidirectional/while_1/Const*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
µ
&bidirectional/while_1/ReadVariableOp_9ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@

+bidirectional/while_1/strided_slice_9/stackConst^bidirectional/while_1/Identity*
valueB"    @   *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_9/stack_1Const^bidirectional/while_1/Identity*
valueB"       *
dtype0*
_output_shapes
:

-bidirectional/while_1/strided_slice_9/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:
À
%bidirectional/while_1/strided_slice_9StridedSlice&bidirectional/while_1/ReadVariableOp_9+bidirectional/while_1/strided_slice_9/stack-bidirectional/while_1/strided_slice_9/stack_1-bidirectional/while_1/strided_slice_9/stack_2*

begin_mask*
Index0*
T0*
end_mask*
_output_shapes

:@@
£
bidirectional/while_1/MatMul_5MatMul bidirectional/while_1/Identity_2%bidirectional/while_1/strided_slice_9*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/add_2Addbidirectional/while_1/BiasAdd_1bidirectional/while_1/MatMul_5*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/mul_1/xConst^bidirectional/while_1/Identity*
valueB
 *ÍÌL>*
dtype0*
_output_shapes
: 

bidirectional/while_1/mul_1Mulbidirectional/while_1/mul_1/xbidirectional/while_1/add_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/add_3/yConst^bidirectional/while_1/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while_1/add_3Addbidirectional/while_1/mul_1bidirectional/while_1/add_3/y*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/Const_2Const^bidirectional/while_1/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while_1/Const_3Const^bidirectional/while_1/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¦
-bidirectional/while_1/clip_by_value_1/MinimumMinimumbidirectional/while_1/add_3bidirectional/while_1/Const_3*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
°
%bidirectional/while_1/clip_by_value_1Maximum-bidirectional/while_1/clip_by_value_1/Minimumbidirectional/while_1/Const_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/mul_2Mul%bidirectional/while_1/clip_by_value_1 bidirectional/while_1/Identity_3*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
¶
'bidirectional/while_1/ReadVariableOp_10ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@

,bidirectional/while_1/strided_slice_10/stackConst^bidirectional/while_1/Identity*
_output_shapes
:*
valueB"       *
dtype0
 
.bidirectional/while_1/strided_slice_10/stack_1Const^bidirectional/while_1/Identity*
valueB"    À   *
dtype0*
_output_shapes
:
 
.bidirectional/while_1/strided_slice_10/stack_2Const^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:*
valueB"      
Å
&bidirectional/while_1/strided_slice_10StridedSlice'bidirectional/while_1/ReadVariableOp_10,bidirectional/while_1/strided_slice_10/stack.bidirectional/while_1/strided_slice_10/stack_1.bidirectional/while_1/strided_slice_10/stack_2*
_output_shapes

:@@*
T0*
Index0*

begin_mask*
end_mask
¤
bidirectional/while_1/MatMul_6MatMul bidirectional/while_1/Identity_2&bidirectional/while_1/strided_slice_10*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/add_4Addbidirectional/while_1/BiasAdd_2bidirectional/while_1/MatMul_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
q
bidirectional/while_1/TanhTanhbidirectional/while_1/add_4*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/mul_3Mul#bidirectional/while_1/clip_by_valuebidirectional/while_1/Tanh*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/add_5Addbidirectional/while_1/mul_2bidirectional/while_1/mul_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
¶
'bidirectional/while_1/ReadVariableOp_11ReadVariableOp,bidirectional/while_1/ReadVariableOp_8/Enter^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:	@

,bidirectional/while_1/strided_slice_11/stackConst^bidirectional/while_1/Identity*
dtype0*
_output_shapes
:*
valueB"    À   
 
.bidirectional/while_1/strided_slice_11/stack_1Const^bidirectional/while_1/Identity*
valueB"        *
dtype0*
_output_shapes
:
 
.bidirectional/while_1/strided_slice_11/stack_2Const^bidirectional/while_1/Identity*
valueB"      *
dtype0*
_output_shapes
:
Å
&bidirectional/while_1/strided_slice_11StridedSlice'bidirectional/while_1/ReadVariableOp_11,bidirectional/while_1/strided_slice_11/stack.bidirectional/while_1/strided_slice_11/stack_1.bidirectional/while_1/strided_slice_11/stack_2*
T0*
Index0*

begin_mask*
end_mask*
_output_shapes

:@@
¤
bidirectional/while_1/MatMul_7MatMul bidirectional/while_1/Identity_2&bidirectional/while_1/strided_slice_11*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/add_6Addbidirectional/while_1/BiasAdd_3bidirectional/while_1/MatMul_7*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/mul_4/xConst^bidirectional/while_1/Identity*
_output_shapes
: *
valueB
 *ÍÌL>*
dtype0

bidirectional/while_1/mul_4Mulbidirectional/while_1/mul_4/xbidirectional/while_1/add_6*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/add_7/yConst^bidirectional/while_1/Identity*
valueB
 *   ?*
dtype0*
_output_shapes
: 

bidirectional/while_1/add_7Addbidirectional/while_1/mul_4bidirectional/while_1/add_7/y*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

bidirectional/while_1/Const_4Const^bidirectional/while_1/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

bidirectional/while_1/Const_5Const^bidirectional/while_1/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¦
-bidirectional/while_1/clip_by_value_2/MinimumMinimumbidirectional/while_1/add_7bidirectional/while_1/Const_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
°
%bidirectional/while_1/clip_by_value_2Maximum-bidirectional/while_1/clip_by_value_2/Minimumbidirectional/while_1/Const_4*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
s
bidirectional/while_1/Tanh_1Tanhbidirectional/while_1/add_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

bidirectional/while_1/mul_5Mul%bidirectional/while_1/clip_by_value_2bidirectional/while_1/Tanh_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
À
9bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3?bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/Enterbidirectional/while_1/Identitybidirectional/while_1/mul_5 bidirectional/while_1/Identity_1*.
_class$
" loc:@bidirectional/while_1/mul_5*
_output_shapes
: *
T0

?bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/EnterEnterbidirectional/TensorArray_3*
is_constant(*
_output_shapes
:*3

frame_name%#bidirectional/while_1/while_context*
T0*.
_class$
" loc:@bidirectional/while_1/mul_5*
parallel_iterations 

bidirectional/while_1/add_8/yConst^bidirectional/while_1/Identity*
_output_shapes
: *
value	B :*
dtype0

bidirectional/while_1/add_8Addbidirectional/while_1/Identitybidirectional/while_1/add_8/y*
T0*
_output_shapes
: 
r
#bidirectional/while_1/NextIterationNextIterationbidirectional/while_1/add_8*
_output_shapes
: *
T0

%bidirectional/while_1/NextIteration_1NextIteration9bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

%bidirectional/while_1/NextIteration_2NextIterationbidirectional/while_1/mul_5*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0

%bidirectional/while_1/NextIteration_3NextIterationbidirectional/while_1/add_5*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
a
bidirectional/while_1/ExitExitbidirectional/while_1/Switch*
_output_shapes
: *
T0
e
bidirectional/while_1/Exit_1Exitbidirectional/while_1/Switch_1*
T0*
_output_shapes
: 
v
bidirectional/while_1/Exit_2Exitbidirectional/while_1/Switch_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
v
bidirectional/while_1/Exit_3Exitbidirectional/while_1/Switch_3*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
Ê
2bidirectional/TensorArrayStack_1/TensorArraySizeV3TensorArraySizeV3bidirectional/TensorArray_3bidirectional/while_1/Exit_1*.
_class$
" loc:@bidirectional/TensorArray_3*
_output_shapes
: 

,bidirectional/TensorArrayStack_1/range/startConst*
value	B : *.
_class$
" loc:@bidirectional/TensorArray_3*
dtype0*
_output_shapes
: 

,bidirectional/TensorArrayStack_1/range/deltaConst*
value	B :*.
_class$
" loc:@bidirectional/TensorArray_3*
dtype0*
_output_shapes
: 

&bidirectional/TensorArrayStack_1/rangeRange,bidirectional/TensorArrayStack_1/range/start2bidirectional/TensorArrayStack_1/TensorArraySizeV3,bidirectional/TensorArrayStack_1/range/delta*.
_class$
" loc:@bidirectional/TensorArray_3*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ç
4bidirectional/TensorArrayStack_1/TensorArrayGatherV3TensorArrayGatherV3bidirectional/TensorArray_3&bidirectional/TensorArrayStack_1/rangebidirectional/while_1/Exit_1*
dtype0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
element_shape:ÿÿÿÿÿÿÿÿÿ@*.
_class$
" loc:@bidirectional/TensorArray_3
w
$bidirectional/strided_slice_31/stackConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_31/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&bidirectional/strided_slice_31/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
±
bidirectional/strided_slice_31StridedSlice4bidirectional/TensorArrayStack_1/TensorArrayGatherV3$bidirectional/strided_slice_31/stack&bidirectional/strided_slice_31/stack_1&bidirectional/strided_slice_31/stack_2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_mask*
Index0*
T0
s
bidirectional/transpose_3/permConst*!
valueB"          *
dtype0*
_output_shapes
:
»
bidirectional/transpose_3	Transpose4bidirectional/TensorArrayStack_1/TensorArrayGatherV3bidirectional/transpose_3/perm*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
[
bidirectional/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
·
bidirectional/concatConcatV2bidirectional/strided_slice_15bidirectional/strided_slice_31bidirectional/concat/axis*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ó5¾*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ó5>*
_class
loc:@dense/kernel*
dtype0
Í
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	@
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	@*
T0*
_class
loc:@dense/kernel
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	@*
T0*
_class
loc:@dense/kernel

dense/kernelVarHandleOp*
shared_namedense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: *
shape:	@
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 

dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0

 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	@

dense/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:@


dense/biasVarHandleOp*
_output_shapes
: *
shape:@*
shared_name
dense/bias*
_class
loc:@dense/bias*
dtype0
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
_class
loc:@dense/bias

dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@*
_class
loc:@dense/bias
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	@
{
dense/MatMulMatMulbidirectional/concatdense/MatMul/ReadVariableOp*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
T0
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
S

dense/ReluReludense/BiasAdd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *¾*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *>*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ò
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
è
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*!
_class
loc:@dense_1/kernel
Ú
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*!
_class
loc:@dense_1/kernel

dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: *
shape
:@
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 

dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
dtype0

"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@

dense_1/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: *
shape:
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 

dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
dtype0

 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
u
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
T0
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
dense_1/SigmoidSigmoiddense_1/BiasAdd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
predict/group_depsNoOp^dense_1/Sigmoid
U
ConstConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_3Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
W
Const_4Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
W
Const_5Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_7Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_9Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
]
Const_10Const"/device:CPU:0*
_output_shapes
: *
valueB Bmodel*
dtype0
X
Const_11Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_12Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_13Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_14Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_15Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_16Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 

RestoreV2/tensor_namesConst*O
valueFBDB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
c
RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

	RestoreV2	RestoreV2Const_10RestoreV2/tensor_namesRestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
Q
AssignVariableOpAssignVariableOpembedding/embeddingsIdentity*
dtype0
X
Const_17Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_18Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 

RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:*K
valueBB@B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
e
RestoreV2_1/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_1	RestoreV2Const_10RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_1IdentityRestoreV2_1*
_output_shapes
:*
T0
M
AssignVariableOp_1AssignVariableOpdense/kernel
Identity_1*
dtype0

RestoreV2_2/tensor_namesConst*I
value@B>B4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_2	RestoreV2Const_10RestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
K
AssignVariableOp_2AssignVariableOp
dense/bias
Identity_2*
dtype0

RestoreV2_3/tensor_namesConst*K
valueBB@B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_3/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_3	RestoreV2Const_10RestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_3IdentityRestoreV2_3*
_output_shapes
:*
T0
O
AssignVariableOp_3AssignVariableOpdense_1/kernel
Identity_3*
dtype0

RestoreV2_4/tensor_namesConst*I
value@B>B4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_4/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_4	RestoreV2Const_10RestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_4IdentityRestoreV2_4*
_output_shapes
:*
T0
M
AssignVariableOp_4AssignVariableOpdense_1/bias
Identity_4*
dtype0
X
Const_19Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_20Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
ª
RestoreV2_5/tensor_namesConst*^
valueUBSBIlayer_with_weights-1/forward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_5	RestoreV2Const_10RestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
b
AssignVariableOp_5AssignVariableOp!bidirectional/forward_lstm/kernel
Identity_5*
dtype0
´
RestoreV2_6/tensor_namesConst*h
value_B]BSlayer_with_weights-1/forward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_6	RestoreV2Const_10RestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_6IdentityRestoreV2_6*
_output_shapes
:*
T0
l
AssignVariableOp_6AssignVariableOp+bidirectional/forward_lstm/recurrent_kernel
Identity_6*
dtype0
¨
RestoreV2_7/tensor_namesConst*\
valueSBQBGlayer_with_weights-1/forward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_7	RestoreV2Const_10RestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_7IdentityRestoreV2_7*
T0*
_output_shapes
:
`
AssignVariableOp_7AssignVariableOpbidirectional/forward_lstm/bias
Identity_7*
dtype0
«
RestoreV2_8/tensor_namesConst*
dtype0*
_output_shapes
:*_
valueVBTBJlayer_with_weights-1/backward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE
e
RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0

RestoreV2_8	RestoreV2Const_10RestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_8IdentityRestoreV2_8*
_output_shapes
:*
T0
c
AssignVariableOp_8AssignVariableOp"bidirectional/backward_lstm/kernel
Identity_8*
dtype0
µ
RestoreV2_9/tensor_namesConst*i
value`B^BTlayer_with_weights-1/backward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_9	RestoreV2Const_10RestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_9IdentityRestoreV2_9*
T0*
_output_shapes
:
m
AssignVariableOp_9AssignVariableOp,bidirectional/backward_lstm/recurrent_kernel
Identity_9*
dtype0
ª
RestoreV2_10/tensor_namesConst*
_output_shapes
:*]
valueTBRBHlayer_with_weights-1/backward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
f
RestoreV2_10/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_10	RestoreV2Const_10RestoreV2_10/tensor_namesRestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
H
Identity_10IdentityRestoreV2_10*
T0*
_output_shapes
:
c
AssignVariableOp_10AssignVariableOp bidirectional/backward_lstm/biasIdentity_10*
dtype0
m
VarIsInitializedOpVarIsInitializedOp+bidirectional/forward_lstm/recurrent_kernel*
_output_shapes
: 
X
VarIsInitializedOp_1VarIsInitializedOpembedding/embeddings*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense_1/bias*
_output_shapes
: 
c
VarIsInitializedOp_3VarIsInitializedOpbidirectional/forward_lstm/bias*
_output_shapes
: 
d
VarIsInitializedOp_4VarIsInitializedOp bidirectional/backward_lstm/bias*
_output_shapes
: 
R
VarIsInitializedOp_5VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
N
VarIsInitializedOp_6VarIsInitializedOp
dense/bias*
_output_shapes
: 
f
VarIsInitializedOp_7VarIsInitializedOp"bidirectional/backward_lstm/kernel*
_output_shapes
: 
e
VarIsInitializedOp_8VarIsInitializedOp!bidirectional/forward_lstm/kernel*
_output_shapes
: 
P
VarIsInitializedOp_9VarIsInitializedOpdense/kernel*
_output_shapes
: 
q
VarIsInitializedOp_10VarIsInitializedOp,bidirectional/backward_lstm/recurrent_kernel*
_output_shapes
: 

initNoOp(^bidirectional/backward_lstm/bias/Assign*^bidirectional/backward_lstm/kernel/Assign4^bidirectional/backward_lstm/recurrent_kernel/Assign'^bidirectional/forward_lstm/bias/Assign)^bidirectional/forward_lstm/kernel/Assign3^bidirectional/forward_lstm/recurrent_kernel/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^embedding/embeddings/Assign
X
Const_21Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
X
Const_22Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
¬
SaveV2/tensor_namesConst"/device:CPU:0*Õ

valueË
BÈ
B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/forward_layer/.ATTRIBUTES/OBJECT_CONFIG_JSONBBlayer_with_weights-1/backward_layer/.ATTRIBUTES/OBJECT_CONFIG_JSONB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-1/forward_layer/cell/.ATTRIBUTES/OBJECT_CONFIG_JSONBGlayer_with_weights-1/backward_layer/cell/.ATTRIBUTES/OBJECT_CONFIG_JSONBIlayer_with_weights-1/forward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/forward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/forward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUEBJlayer_with_weights-1/backward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/backward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-1/backward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:

SaveV2/shape_and_slicesConst"/device:CPU:0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ú
SaveV2SaveV2Const_22SaveV2/tensor_namesSaveV2/shape_and_slicesConst_11Const_12Const_13Const_14Const_15Const_16(embedding/embeddings/Read/ReadVariableOpConst_17Const_18 dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst_19Const_205bidirectional/forward_lstm/kernel/Read/ReadVariableOp?bidirectional/forward_lstm/recurrent_kernel/Read/ReadVariableOp3bidirectional/forward_lstm/bias/Read/ReadVariableOp6bidirectional/backward_lstm/kernel/Read/ReadVariableOp@bidirectional/backward_lstm/recurrent_kernel/Read/ReadVariableOp4bidirectional/backward_lstm/bias/Read/ReadVariableOpConst_21"/device:CPU:0*$
dtypes
2
Z
Identity_11IdentityConst_22^SaveV2"/device:CPU:0*
T0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 

save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*·

value­
Bª
B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONBBlayer_with_weights-1/backward_layer/.ATTRIBUTES/OBJECT_CONFIG_JSONBGlayer_with_weights-1/backward_layer/cell/.ATTRIBUTES/OBJECT_CONFIG_JSONBHlayer_with_weights-1/backward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUEBJlayer_with_weights-1/backward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/backward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/forward_layer/.ATTRIBUTES/OBJECT_CONFIG_JSONBFlayer_with_weights-1/forward_layer/cell/.ATTRIBUTES/OBJECT_CONFIG_JSONBGlayer_with_weights-1/forward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/forward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/forward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE

save/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
¡
save/SaveV2/tensors_0Const*Û
valueÑBÎ BÇ{"class_name": "Sequential", "config": {"layers": [{"class_name": "Embedding", "config": {"activity_regularizer": null, "batch_input_shape": [null, null], "dtype": "float32", "embeddings_constraint": null, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"dtype": "float32", "maxval": 0.05, "minval": -0.05, "seed": null}}, "embeddings_regularizer": null, "input_dim": 8185, "input_length": null, "mask_zero": false, "name": "embedding", "output_dim": 64, "trainable": true}}, {"class_name": "Bidirectional", "config": {"dtype": "float32", "layer": {"class_name": "LSTM", "config": {"activation": "tanh", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dropout": 0.0, "dtype": null, "go_backwards": false, "implementation": 1, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "lstm", "recurrent_activation": "hard_sigmoid", "recurrent_constraint": null, "recurrent_dropout": 0.0, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"dtype": "float32", "gain": 1.0, "seed": null}}, "recurrent_regularizer": null, "return_sequences": false, "return_state": false, "stateful": false, "time_major": false, "trainable": true, "unit_forget_bias": true, "units": 64, "unroll": false, "use_bias": true, "zero_output_for_mask": true}}, "merge_mode": "concat", "name": "bidirectional", "trainable": true}}, {"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense", "trainable": true, "units": 64, "use_bias": true}}, {"class_name": "Dense", "config": {"activation": "sigmoid", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 1, "use_bias": true}}], "name": "sequential"}}*
dtype0*
_output_shapes
: 
å
save/SaveV2/tensors_1Const*
dtype0*
_output_shapes
: *
valueB B{"class_name": "InputLayer", "config": {"batch_input_shape": [null, null], "dtype": "float32", "name": "embedding_input", "sparse": false}}

save/SaveV2/tensors_2Const*Ð
valueÆBÃ B¼{"class_name": "Embedding", "config": {"activity_regularizer": null, "batch_input_shape": [null, null], "dtype": "float32", "embeddings_constraint": null, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"dtype": "float32", "maxval": 0.05, "minval": -0.05, "seed": null}}, "embeddings_regularizer": null, "input_dim": 8185, "input_length": null, "mask_zero": false, "name": "embedding", "output_dim": 64, "trainable": true}}*
dtype0*
_output_shapes
: 
û
save/SaveV2/tensors_4Const*µ
value«B¨ B¡{"class_name": "Bidirectional", "config": {"dtype": "float32", "layer": {"class_name": "LSTM", "config": {"activation": "tanh", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dropout": 0.0, "dtype": null, "go_backwards": false, "implementation": 1, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "lstm", "recurrent_activation": "hard_sigmoid", "recurrent_constraint": null, "recurrent_dropout": 0.0, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"dtype": "float32", "gain": 1.0, "seed": null}}, "recurrent_regularizer": null, "return_sequences": false, "return_state": false, "stateful": false, "time_major": false, "trainable": true, "unit_forget_bias": true, "units": 64, "unroll": false, "use_bias": true, "zero_output_for_mask": true}}, "merge_mode": "concat", "name": "bidirectional", "trainable": true}}*
dtype0*
_output_shapes
: 
õ
save/SaveV2/tensors_5Const*
_output_shapes
: *¯
value¥B¢ B{"class_name": "LSTM", "config": {"activation": "tanh", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dropout": 0.0, "dtype": null, "go_backwards": true, "implementation": 1, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "backward_lstm", "recurrent_activation": "hard_sigmoid", "recurrent_constraint": null, "recurrent_dropout": 0.0, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"dtype": "float32", "gain": 1.0, "seed": null}}, "recurrent_regularizer": null, "return_sequences": false, "return_state": false, "stateful": false, "time_major": false, "trainable": true, "unit_forget_bias": true, "units": 64, "unroll": false, "use_bias": true, "zero_output_for_mask": true}}*
dtype0
º
save/SaveV2/tensors_6Const*ô
valueêBç Bà{"class_name": "LSTMCell", "config": {"activation": "tanh", "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dropout": 0.0, "dtype": null, "implementation": 1, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "lstm_cell_1", "recurrent_activation": "hard_sigmoid", "recurrent_constraint": null, "recurrent_dropout": 0.0, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"dtype": "float32", "gain": 1.0, "seed": null}}, "recurrent_regularizer": null, "trainable": true, "unit_forget_bias": true, "units": 64, "use_bias": true}}*
dtype0*
_output_shapes
: 
ö
save/SaveV2/tensors_10Const*
_output_shapes
: *¯
value¥B¢ B{"class_name": "LSTM", "config": {"activation": "tanh", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dropout": 0.0, "dtype": null, "go_backwards": false, "implementation": 1, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "forward_lstm", "recurrent_activation": "hard_sigmoid", "recurrent_constraint": null, "recurrent_dropout": 0.0, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"dtype": "float32", "gain": 1.0, "seed": null}}, "recurrent_regularizer": null, "return_sequences": false, "return_state": false, "stateful": false, "time_major": false, "trainable": true, "unit_forget_bias": true, "units": 64, "unroll": false, "use_bias": true, "zero_output_for_mask": true}}*
dtype0
¹
save/SaveV2/tensors_11Const*ò
valueèBå BÞ{"class_name": "LSTMCell", "config": {"activation": "tanh", "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dropout": 0.0, "dtype": null, "implementation": 1, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "lstm_cell", "recurrent_activation": "hard_sigmoid", "recurrent_constraint": null, "recurrent_dropout": 0.0, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"dtype": "float32", "gain": 1.0, "seed": null}}, "recurrent_regularizer": null, "trainable": true, "unit_forget_bias": true, "units": 64, "use_bias": true}}*
dtype0*
_output_shapes
: 
¥
save/SaveV2/tensors_15Const*Þ
valueÔBÑ BÊ{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense", "trainable": true, "units": 64, "use_bias": true}}*
dtype0*
_output_shapes
: 
©
save/SaveV2/tensors_18Const*â
valueØBÕ BÎ{"class_name": "Dense", "config": {"activation": "sigmoid", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 1, "use_bias": true}}*
dtype0*
_output_shapes
: 
÷
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/SaveV2/tensors_0save/SaveV2/tensors_1save/SaveV2/tensors_2(embedding/embeddings/Read/ReadVariableOpsave/SaveV2/tensors_4save/SaveV2/tensors_5save/SaveV2/tensors_64bidirectional/backward_lstm/bias/Read/ReadVariableOp6bidirectional/backward_lstm/kernel/Read/ReadVariableOp@bidirectional/backward_lstm/recurrent_kernel/Read/ReadVariableOpsave/SaveV2/tensors_10save/SaveV2/tensors_113bidirectional/forward_lstm/bias/Read/ReadVariableOp5bidirectional/forward_lstm/kernel/Read/ReadVariableOp?bidirectional/forward_lstm/recurrent_kernel/Read/ReadVariableOpsave/SaveV2/tensors_15dense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpsave/SaveV2/tensors_18 dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp*#
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*·

value­
Bª
B/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONBBlayer_with_weights-1/backward_layer/.ATTRIBUTES/OBJECT_CONFIG_JSONBGlayer_with_weights-1/backward_layer/cell/.ATTRIBUTES/OBJECT_CONFIG_JSONBHlayer_with_weights-1/backward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUEBJlayer_with_weights-1/backward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/backward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/forward_layer/.ATTRIBUTES/OBJECT_CONFIG_JSONBFlayer_with_weights-1/forward_layer/cell/.ATTRIBUTES/OBJECT_CONFIG_JSONBGlayer_with_weights-1/forward_layer/cell/bias/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/forward_layer/cell/kernel/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/forward_layer/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp
N
save/IdentityIdentitysave/RestoreV2:3*
T0*
_output_shapes
:
[
save/AssignVariableOpAssignVariableOpembedding/embeddingssave/Identity*
dtype0

save/NoOp_3NoOp

save/NoOp_4NoOp

save/NoOp_5NoOp
P
save/Identity_1Identitysave/RestoreV2:7*
T0*
_output_shapes
:
k
save/AssignVariableOp_1AssignVariableOp bidirectional/backward_lstm/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:8*
_output_shapes
:*
T0
m
save/AssignVariableOp_2AssignVariableOp"bidirectional/backward_lstm/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:9*
_output_shapes
:*
T0
w
save/AssignVariableOp_3AssignVariableOp,bidirectional/backward_lstm/recurrent_kernelsave/Identity_3*
dtype0

save/NoOp_6NoOp

save/NoOp_7NoOp
Q
save/Identity_4Identitysave/RestoreV2:12*
T0*
_output_shapes
:
j
save/AssignVariableOp_4AssignVariableOpbidirectional/forward_lstm/biassave/Identity_4*
dtype0
Q
save/Identity_5Identitysave/RestoreV2:13*
_output_shapes
:*
T0
l
save/AssignVariableOp_5AssignVariableOp!bidirectional/forward_lstm/kernelsave/Identity_5*
dtype0
Q
save/Identity_6Identitysave/RestoreV2:14*
_output_shapes
:*
T0
v
save/AssignVariableOp_6AssignVariableOp+bidirectional/forward_lstm/recurrent_kernelsave/Identity_6*
dtype0

save/NoOp_8NoOp
Q
save/Identity_7Identitysave/RestoreV2:16*
T0*
_output_shapes
:
U
save/AssignVariableOp_7AssignVariableOp
dense/biassave/Identity_7*
dtype0
Q
save/Identity_8Identitysave/RestoreV2:17*
_output_shapes
:*
T0
W
save/AssignVariableOp_8AssignVariableOpdense/kernelsave/Identity_8*
dtype0

save/NoOp_9NoOp
Q
save/Identity_9Identitysave/RestoreV2:19*
T0*
_output_shapes
:
W
save/AssignVariableOp_9AssignVariableOpdense_1/biassave/Identity_9*
dtype0
R
save/Identity_10Identitysave/RestoreV2:20*
T0*
_output_shapes
:
[
save/AssignVariableOp_10AssignVariableOpdense_1/kernelsave/Identity_10*
dtype0
¿
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
^save/NoOp^save/NoOp_1^save/NoOp_2^save/NoOp_3^save/NoOp_4^save/NoOp_5^save/NoOp_6^save/NoOp_7^save/NoOp_8^save/NoOp_9

init_1NoOp"D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"
trainable_variablesùö

embedding/embeddings:0embedding/embeddings/Assign*embedding/embeddings/Read/ReadVariableOp:0(21embedding/embeddings/Initializer/random_uniform:08
Ì
#bidirectional/forward_lstm/kernel:0(bidirectional/forward_lstm/kernel/Assign7bidirectional/forward_lstm/kernel/Read/ReadVariableOp:0(2>bidirectional/forward_lstm/kernel/Initializer/random_uniform:08
ë
-bidirectional/forward_lstm/recurrent_kernel:02bidirectional/forward_lstm/recurrent_kernel/AssignAbidirectional/forward_lstm/recurrent_kernel/Read/ReadVariableOp:0(2?bidirectional/forward_lstm/recurrent_kernel/Initializer/mul_1:08
¼
!bidirectional/forward_lstm/bias:0&bidirectional/forward_lstm/bias/Assign5bidirectional/forward_lstm/bias/Read/ReadVariableOp:0(24bidirectional/forward_lstm/bias/Initializer/concat:08
Ð
$bidirectional/backward_lstm/kernel:0)bidirectional/backward_lstm/kernel/Assign8bidirectional/backward_lstm/kernel/Read/ReadVariableOp:0(2?bidirectional/backward_lstm/kernel/Initializer/random_uniform:08
ï
.bidirectional/backward_lstm/recurrent_kernel:03bidirectional/backward_lstm/recurrent_kernel/AssignBbidirectional/backward_lstm/recurrent_kernel/Read/ReadVariableOp:0(2@bidirectional/backward_lstm/recurrent_kernel/Initializer/mul_1:08
À
"bidirectional/backward_lstm/bias:0'bidirectional/backward_lstm/bias/Assign6bidirectional/backward_lstm/bias/Read/ReadVariableOp:0(25bidirectional/backward_lstm/bias/Initializer/concat:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"
	variablesùö

embedding/embeddings:0embedding/embeddings/Assign*embedding/embeddings/Read/ReadVariableOp:0(21embedding/embeddings/Initializer/random_uniform:08
Ì
#bidirectional/forward_lstm/kernel:0(bidirectional/forward_lstm/kernel/Assign7bidirectional/forward_lstm/kernel/Read/ReadVariableOp:0(2>bidirectional/forward_lstm/kernel/Initializer/random_uniform:08
ë
-bidirectional/forward_lstm/recurrent_kernel:02bidirectional/forward_lstm/recurrent_kernel/AssignAbidirectional/forward_lstm/recurrent_kernel/Read/ReadVariableOp:0(2?bidirectional/forward_lstm/recurrent_kernel/Initializer/mul_1:08
¼
!bidirectional/forward_lstm/bias:0&bidirectional/forward_lstm/bias/Assign5bidirectional/forward_lstm/bias/Read/ReadVariableOp:0(24bidirectional/forward_lstm/bias/Initializer/concat:08
Ð
$bidirectional/backward_lstm/kernel:0)bidirectional/backward_lstm/kernel/Assign8bidirectional/backward_lstm/kernel/Read/ReadVariableOp:0(2?bidirectional/backward_lstm/kernel/Initializer/random_uniform:08
ï
.bidirectional/backward_lstm/recurrent_kernel:03bidirectional/backward_lstm/recurrent_kernel/AssignBbidirectional/backward_lstm/recurrent_kernel/Read/ReadVariableOp:0(2@bidirectional/backward_lstm/recurrent_kernel/Initializer/mul_1:08
À
"bidirectional/backward_lstm/bias:0'bidirectional/backward_lstm/bias/Assign6bidirectional/backward_lstm/bias/Read/ReadVariableOp:0(25bidirectional/backward_lstm/bias/Initializer/concat:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"Öo
while_contextÄoÁo
«6
!bidirectional/while/while_context  *bidirectional/while/LoopCond:02bidirectional/while/Merge:0:bidirectional/while/Identity:0Bbidirectional/while/Exit:0Bbidirectional/while/Exit_1:0Bbidirectional/while/Exit_2:0Bbidirectional/while/Exit_3:0J²3
bidirectional/TensorArray:0
Jbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
bidirectional/TensorArray_1:0
!bidirectional/forward_lstm/bias:0
#bidirectional/forward_lstm/kernel:0
-bidirectional/forward_lstm/recurrent_kernel:0
bidirectional/strided_slice_1:0
bidirectional/while/BiasAdd:0
bidirectional/while/BiasAdd_1:0
bidirectional/while/BiasAdd_2:0
bidirectional/while/BiasAdd_3:0
bidirectional/while/Const:0
bidirectional/while/Const_1:0
bidirectional/while/Const_2:0
bidirectional/while/Const_3:0
bidirectional/while/Const_4:0
bidirectional/while/Const_5:0
bidirectional/while/Enter:0
bidirectional/while/Enter_1:0
bidirectional/while/Enter_2:0
bidirectional/while/Enter_3:0
bidirectional/while/Exit:0
bidirectional/while/Exit_1:0
bidirectional/while/Exit_2:0
bidirectional/while/Exit_3:0
bidirectional/while/Identity:0
 bidirectional/while/Identity_1:0
 bidirectional/while/Identity_2:0
 bidirectional/while/Identity_3:0
 bidirectional/while/Less/Enter:0
bidirectional/while/Less:0
bidirectional/while/LoopCond:0
bidirectional/while/MatMul:0
bidirectional/while/MatMul_1:0
bidirectional/while/MatMul_2:0
bidirectional/while/MatMul_3:0
bidirectional/while/MatMul_4:0
bidirectional/while/MatMul_5:0
bidirectional/while/MatMul_6:0
bidirectional/while/MatMul_7:0
bidirectional/while/Merge:0
bidirectional/while/Merge:1
bidirectional/while/Merge_1:0
bidirectional/while/Merge_1:1
bidirectional/while/Merge_2:0
bidirectional/while/Merge_2:1
bidirectional/while/Merge_3:0
bidirectional/while/Merge_3:1
#bidirectional/while/NextIteration:0
%bidirectional/while/NextIteration_1:0
%bidirectional/while/NextIteration_2:0
%bidirectional/while/NextIteration_3:0
*bidirectional/while/ReadVariableOp/Enter:0
$bidirectional/while/ReadVariableOp:0
'bidirectional/while/ReadVariableOp_10:0
'bidirectional/while/ReadVariableOp_11:0
&bidirectional/while/ReadVariableOp_1:0
&bidirectional/while/ReadVariableOp_2:0
&bidirectional/while/ReadVariableOp_3:0
,bidirectional/while/ReadVariableOp_4/Enter:0
&bidirectional/while/ReadVariableOp_4:0
&bidirectional/while/ReadVariableOp_5:0
&bidirectional/while/ReadVariableOp_6:0
&bidirectional/while/ReadVariableOp_7:0
,bidirectional/while/ReadVariableOp_8/Enter:0
&bidirectional/while/ReadVariableOp_8:0
&bidirectional/while/ReadVariableOp_9:0
bidirectional/while/Switch:0
bidirectional/while/Switch:1
bidirectional/while/Switch_1:0
bidirectional/while/Switch_1:1
bidirectional/while/Switch_2:0
bidirectional/while/Switch_2:1
bidirectional/while/Switch_3:0
bidirectional/while/Switch_3:1
bidirectional/while/Tanh:0
bidirectional/while/Tanh_1:0
-bidirectional/while/TensorArrayReadV3/Enter:0
/bidirectional/while/TensorArrayReadV3/Enter_1:0
'bidirectional/while/TensorArrayReadV3:0
?bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
9bidirectional/while/TensorArrayWrite/TensorArrayWriteV3:0
bidirectional/while/add:0
bidirectional/while/add_1/y:0
bidirectional/while/add_1:0
bidirectional/while/add_2:0
bidirectional/while/add_3/y:0
bidirectional/while/add_3:0
bidirectional/while/add_4:0
bidirectional/while/add_5:0
bidirectional/while/add_6:0
bidirectional/while/add_7/y:0
bidirectional/while/add_7:0
bidirectional/while/add_8/y:0
bidirectional/while/add_8:0
+bidirectional/while/clip_by_value/Minimum:0
#bidirectional/while/clip_by_value:0
-bidirectional/while/clip_by_value_1/Minimum:0
%bidirectional/while/clip_by_value_1:0
-bidirectional/while/clip_by_value_2/Minimum:0
%bidirectional/while/clip_by_value_2:0
bidirectional/while/mul/x:0
bidirectional/while/mul:0
bidirectional/while/mul_1/x:0
bidirectional/while/mul_1:0
bidirectional/while/mul_2:0
bidirectional/while/mul_3:0
bidirectional/while/mul_4/x:0
bidirectional/while/mul_4:0
bidirectional/while/mul_5:0
)bidirectional/while/strided_slice/stack:0
+bidirectional/while/strided_slice/stack_1:0
+bidirectional/while/strided_slice/stack_2:0
#bidirectional/while/strided_slice:0
+bidirectional/while/strided_slice_1/stack:0
-bidirectional/while/strided_slice_1/stack_1:0
-bidirectional/while/strided_slice_1/stack_2:0
,bidirectional/while/strided_slice_10/stack:0
.bidirectional/while/strided_slice_10/stack_1:0
.bidirectional/while/strided_slice_10/stack_2:0
&bidirectional/while/strided_slice_10:0
,bidirectional/while/strided_slice_11/stack:0
.bidirectional/while/strided_slice_11/stack_1:0
.bidirectional/while/strided_slice_11/stack_2:0
&bidirectional/while/strided_slice_11:0
%bidirectional/while/strided_slice_1:0
+bidirectional/while/strided_slice_2/stack:0
-bidirectional/while/strided_slice_2/stack_1:0
-bidirectional/while/strided_slice_2/stack_2:0
%bidirectional/while/strided_slice_2:0
+bidirectional/while/strided_slice_3/stack:0
-bidirectional/while/strided_slice_3/stack_1:0
-bidirectional/while/strided_slice_3/stack_2:0
%bidirectional/while/strided_slice_3:0
+bidirectional/while/strided_slice_4/stack:0
-bidirectional/while/strided_slice_4/stack_1:0
-bidirectional/while/strided_slice_4/stack_2:0
%bidirectional/while/strided_slice_4:0
+bidirectional/while/strided_slice_5/stack:0
-bidirectional/while/strided_slice_5/stack_1:0
-bidirectional/while/strided_slice_5/stack_2:0
%bidirectional/while/strided_slice_5:0
+bidirectional/while/strided_slice_6/stack:0
-bidirectional/while/strided_slice_6/stack_1:0
-bidirectional/while/strided_slice_6/stack_2:0
%bidirectional/while/strided_slice_6:0
+bidirectional/while/strided_slice_7/stack:0
-bidirectional/while/strided_slice_7/stack_1:0
-bidirectional/while/strided_slice_7/stack_2:0
%bidirectional/while/strided_slice_7:0
+bidirectional/while/strided_slice_8/stack:0
-bidirectional/while/strided_slice_8/stack_1:0
-bidirectional/while/strided_slice_8/stack_2:0
%bidirectional/while/strided_slice_8:0
+bidirectional/while/strided_slice_9/stack:0
-bidirectional/while/strided_slice_9/stack_1:0
-bidirectional/while/strided_slice_9/stack_2:0
%bidirectional/while/strided_slice_9:0Q
!bidirectional/forward_lstm/bias:0,bidirectional/while/ReadVariableOp_4/Enter:0]
-bidirectional/forward_lstm/recurrent_kernel:0,bidirectional/while/ReadVariableOp_8/Enter:0`
bidirectional/TensorArray_1:0?bidirectional/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0C
bidirectional/strided_slice_1:0 bidirectional/while/Less/Enter:0}
Jbidirectional/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0/bidirectional/while/TensorArrayReadV3/Enter_1:0L
bidirectional/TensorArray:0-bidirectional/while/TensorArrayReadV3/Enter:0Q
#bidirectional/forward_lstm/kernel:0*bidirectional/while/ReadVariableOp/Enter:0Rbidirectional/while/Enter:0Rbidirectional/while/Enter_1:0Rbidirectional/while/Enter_2:0Rbidirectional/while/Enter_3:0
9
#bidirectional/while_1/while_context  * bidirectional/while_1/LoopCond:02bidirectional/while_1/Merge:0: bidirectional/while_1/Identity:0Bbidirectional/while_1/Exit:0Bbidirectional/while_1/Exit_1:0Bbidirectional/while_1/Exit_2:0Bbidirectional/while_1/Exit_3:0Jÿ5
Lbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3:0
bidirectional/TensorArray_2:0
bidirectional/TensorArray_3:0
"bidirectional/backward_lstm/bias:0
$bidirectional/backward_lstm/kernel:0
.bidirectional/backward_lstm/recurrent_kernel:0
 bidirectional/strided_slice_17:0
bidirectional/while_1/BiasAdd:0
!bidirectional/while_1/BiasAdd_1:0
!bidirectional/while_1/BiasAdd_2:0
!bidirectional/while_1/BiasAdd_3:0
bidirectional/while_1/Const:0
bidirectional/while_1/Const_1:0
bidirectional/while_1/Const_2:0
bidirectional/while_1/Const_3:0
bidirectional/while_1/Const_4:0
bidirectional/while_1/Const_5:0
bidirectional/while_1/Enter:0
bidirectional/while_1/Enter_1:0
bidirectional/while_1/Enter_2:0
bidirectional/while_1/Enter_3:0
bidirectional/while_1/Exit:0
bidirectional/while_1/Exit_1:0
bidirectional/while_1/Exit_2:0
bidirectional/while_1/Exit_3:0
 bidirectional/while_1/Identity:0
"bidirectional/while_1/Identity_1:0
"bidirectional/while_1/Identity_2:0
"bidirectional/while_1/Identity_3:0
"bidirectional/while_1/Less/Enter:0
bidirectional/while_1/Less:0
 bidirectional/while_1/LoopCond:0
bidirectional/while_1/MatMul:0
 bidirectional/while_1/MatMul_1:0
 bidirectional/while_1/MatMul_2:0
 bidirectional/while_1/MatMul_3:0
 bidirectional/while_1/MatMul_4:0
 bidirectional/while_1/MatMul_5:0
 bidirectional/while_1/MatMul_6:0
 bidirectional/while_1/MatMul_7:0
bidirectional/while_1/Merge:0
bidirectional/while_1/Merge:1
bidirectional/while_1/Merge_1:0
bidirectional/while_1/Merge_1:1
bidirectional/while_1/Merge_2:0
bidirectional/while_1/Merge_2:1
bidirectional/while_1/Merge_3:0
bidirectional/while_1/Merge_3:1
%bidirectional/while_1/NextIteration:0
'bidirectional/while_1/NextIteration_1:0
'bidirectional/while_1/NextIteration_2:0
'bidirectional/while_1/NextIteration_3:0
,bidirectional/while_1/ReadVariableOp/Enter:0
&bidirectional/while_1/ReadVariableOp:0
)bidirectional/while_1/ReadVariableOp_10:0
)bidirectional/while_1/ReadVariableOp_11:0
(bidirectional/while_1/ReadVariableOp_1:0
(bidirectional/while_1/ReadVariableOp_2:0
(bidirectional/while_1/ReadVariableOp_3:0
.bidirectional/while_1/ReadVariableOp_4/Enter:0
(bidirectional/while_1/ReadVariableOp_4:0
(bidirectional/while_1/ReadVariableOp_5:0
(bidirectional/while_1/ReadVariableOp_6:0
(bidirectional/while_1/ReadVariableOp_7:0
.bidirectional/while_1/ReadVariableOp_8/Enter:0
(bidirectional/while_1/ReadVariableOp_8:0
(bidirectional/while_1/ReadVariableOp_9:0
bidirectional/while_1/Switch:0
bidirectional/while_1/Switch:1
 bidirectional/while_1/Switch_1:0
 bidirectional/while_1/Switch_1:1
 bidirectional/while_1/Switch_2:0
 bidirectional/while_1/Switch_2:1
 bidirectional/while_1/Switch_3:0
 bidirectional/while_1/Switch_3:1
bidirectional/while_1/Tanh:0
bidirectional/while_1/Tanh_1:0
/bidirectional/while_1/TensorArrayReadV3/Enter:0
1bidirectional/while_1/TensorArrayReadV3/Enter_1:0
)bidirectional/while_1/TensorArrayReadV3:0
Abidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/Enter:0
;bidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3:0
bidirectional/while_1/add:0
bidirectional/while_1/add_1/y:0
bidirectional/while_1/add_1:0
bidirectional/while_1/add_2:0
bidirectional/while_1/add_3/y:0
bidirectional/while_1/add_3:0
bidirectional/while_1/add_4:0
bidirectional/while_1/add_5:0
bidirectional/while_1/add_6:0
bidirectional/while_1/add_7/y:0
bidirectional/while_1/add_7:0
bidirectional/while_1/add_8/y:0
bidirectional/while_1/add_8:0
-bidirectional/while_1/clip_by_value/Minimum:0
%bidirectional/while_1/clip_by_value:0
/bidirectional/while_1/clip_by_value_1/Minimum:0
'bidirectional/while_1/clip_by_value_1:0
/bidirectional/while_1/clip_by_value_2/Minimum:0
'bidirectional/while_1/clip_by_value_2:0
bidirectional/while_1/mul/x:0
bidirectional/while_1/mul:0
bidirectional/while_1/mul_1/x:0
bidirectional/while_1/mul_1:0
bidirectional/while_1/mul_2:0
bidirectional/while_1/mul_3:0
bidirectional/while_1/mul_4/x:0
bidirectional/while_1/mul_4:0
bidirectional/while_1/mul_5:0
+bidirectional/while_1/strided_slice/stack:0
-bidirectional/while_1/strided_slice/stack_1:0
-bidirectional/while_1/strided_slice/stack_2:0
%bidirectional/while_1/strided_slice:0
-bidirectional/while_1/strided_slice_1/stack:0
/bidirectional/while_1/strided_slice_1/stack_1:0
/bidirectional/while_1/strided_slice_1/stack_2:0
.bidirectional/while_1/strided_slice_10/stack:0
0bidirectional/while_1/strided_slice_10/stack_1:0
0bidirectional/while_1/strided_slice_10/stack_2:0
(bidirectional/while_1/strided_slice_10:0
.bidirectional/while_1/strided_slice_11/stack:0
0bidirectional/while_1/strided_slice_11/stack_1:0
0bidirectional/while_1/strided_slice_11/stack_2:0
(bidirectional/while_1/strided_slice_11:0
'bidirectional/while_1/strided_slice_1:0
-bidirectional/while_1/strided_slice_2/stack:0
/bidirectional/while_1/strided_slice_2/stack_1:0
/bidirectional/while_1/strided_slice_2/stack_2:0
'bidirectional/while_1/strided_slice_2:0
-bidirectional/while_1/strided_slice_3/stack:0
/bidirectional/while_1/strided_slice_3/stack_1:0
/bidirectional/while_1/strided_slice_3/stack_2:0
'bidirectional/while_1/strided_slice_3:0
-bidirectional/while_1/strided_slice_4/stack:0
/bidirectional/while_1/strided_slice_4/stack_1:0
/bidirectional/while_1/strided_slice_4/stack_2:0
'bidirectional/while_1/strided_slice_4:0
-bidirectional/while_1/strided_slice_5/stack:0
/bidirectional/while_1/strided_slice_5/stack_1:0
/bidirectional/while_1/strided_slice_5/stack_2:0
'bidirectional/while_1/strided_slice_5:0
-bidirectional/while_1/strided_slice_6/stack:0
/bidirectional/while_1/strided_slice_6/stack_1:0
/bidirectional/while_1/strided_slice_6/stack_2:0
'bidirectional/while_1/strided_slice_6:0
-bidirectional/while_1/strided_slice_7/stack:0
/bidirectional/while_1/strided_slice_7/stack_1:0
/bidirectional/while_1/strided_slice_7/stack_2:0
'bidirectional/while_1/strided_slice_7:0
-bidirectional/while_1/strided_slice_8/stack:0
/bidirectional/while_1/strided_slice_8/stack_1:0
/bidirectional/while_1/strided_slice_8/stack_2:0
'bidirectional/while_1/strided_slice_8:0
-bidirectional/while_1/strided_slice_9/stack:0
/bidirectional/while_1/strided_slice_9/stack_1:0
/bidirectional/while_1/strided_slice_9/stack_2:0
'bidirectional/while_1/strided_slice_9:0F
 bidirectional/strided_slice_17:0"bidirectional/while_1/Less/Enter:0
Lbidirectional/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3:01bidirectional/while_1/TensorArrayReadV3/Enter_1:0`
.bidirectional/backward_lstm/recurrent_kernel:0.bidirectional/while_1/ReadVariableOp_8/Enter:0P
bidirectional/TensorArray_2:0/bidirectional/while_1/TensorArrayReadV3/Enter:0T
$bidirectional/backward_lstm/kernel:0,bidirectional/while_1/ReadVariableOp/Enter:0T
"bidirectional/backward_lstm/bias:0.bidirectional/while_1/ReadVariableOp_4/Enter:0b
bidirectional/TensorArray_3:0Abidirectional/while_1/TensorArrayWrite/TensorArrayWriteV3/Enter:0Rbidirectional/while_1/Enter:0Rbidirectional/while_1/Enter_1:0Rbidirectional/while_1/Enter_2:0Rbidirectional/while_1/Enter_3:0*«
serving_default
D
embedding_input1
embedding_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ3
dense_1(
dense_1/Sigmoid:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*@
__saved_model_init_op'%
__saved_model_init_op
init_1