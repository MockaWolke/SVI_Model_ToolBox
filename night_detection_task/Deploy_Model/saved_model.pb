Þ
´
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68²ø
~
Conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *
shared_nameConv_1/kernel
w
!Conv_1/kernel/Read/ReadVariableOpReadVariableOpConv_1/kernel*&
_output_shapes
:		 *
dtype0
n
Conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_1/bias
g
Conv_1/bias/Read/ReadVariableOpReadVariableOpConv_1/bias*
_output_shapes
: *
dtype0
z
BatchNorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameBatchNorm_1/gamma
s
%BatchNorm_1/gamma/Read/ReadVariableOpReadVariableOpBatchNorm_1/gamma*
_output_shapes
: *
dtype0
x
BatchNorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameBatchNorm_1/beta
q
$BatchNorm_1/beta/Read/ReadVariableOpReadVariableOpBatchNorm_1/beta*
_output_shapes
: *
dtype0

BatchNorm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameBatchNorm_1/moving_mean

+BatchNorm_1/moving_mean/Read/ReadVariableOpReadVariableOpBatchNorm_1/moving_mean*
_output_shapes
: *
dtype0

BatchNorm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameBatchNorm_1/moving_variance

/BatchNorm_1/moving_variance/Read/ReadVariableOpReadVariableOpBatchNorm_1/moving_variance*
_output_shapes
: *
dtype0
~
Conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameConv_2/kernel
w
!Conv_2/kernel/Read/ReadVariableOpReadVariableOpConv_2/kernel*&
_output_shapes
: @*
dtype0
n
Conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv_2/bias
g
Conv_2/bias/Read/ReadVariableOpReadVariableOpConv_2/bias*
_output_shapes
:@*
dtype0
z
BatchNorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameBatchNorm_2/gamma
s
%BatchNorm_2/gamma/Read/ReadVariableOpReadVariableOpBatchNorm_2/gamma*
_output_shapes
:@*
dtype0
x
BatchNorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameBatchNorm_2/beta
q
$BatchNorm_2/beta/Read/ReadVariableOpReadVariableOpBatchNorm_2/beta*
_output_shapes
:@*
dtype0

BatchNorm_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameBatchNorm_2/moving_mean

+BatchNorm_2/moving_mean/Read/ReadVariableOpReadVariableOpBatchNorm_2/moving_mean*
_output_shapes
:@*
dtype0

BatchNorm_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameBatchNorm_2/moving_variance

/BatchNorm_2/moving_variance/Read/ReadVariableOpReadVariableOpBatchNorm_2/moving_variance*
_output_shapes
:@*
dtype0
~
Conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameConv_3/kernel
w
!Conv_3/kernel/Read/ReadVariableOpReadVariableOpConv_3/kernel*&
_output_shapes
:@@*
dtype0
n
Conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv_3/bias
g
Conv_3/bias/Read/ReadVariableOpReadVariableOpConv_3/bias*
_output_shapes
:@*
dtype0
z
BatchNorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameBatchNorm_3/gamma
s
%BatchNorm_3/gamma/Read/ReadVariableOpReadVariableOpBatchNorm_3/gamma*
_output_shapes
:@*
dtype0
x
BatchNorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameBatchNorm_3/beta
q
$BatchNorm_3/beta/Read/ReadVariableOpReadVariableOpBatchNorm_3/beta*
_output_shapes
:@*
dtype0

BatchNorm_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameBatchNorm_3/moving_mean

+BatchNorm_3/moving_mean/Read/ReadVariableOpReadVariableOpBatchNorm_3/moving_mean*
_output_shapes
:@*
dtype0

BatchNorm_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameBatchNorm_3/moving_variance

/BatchNorm_3/moving_variance/Read/ReadVariableOpReadVariableOpBatchNorm_3/moving_variance*
_output_shapes
:@*
dtype0

Predictions/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_namePredictions/kernel
y
&Predictions/kernel/Read/ReadVariableOpReadVariableOpPredictions/kernel*
_output_shapes

:@*
dtype0
x
Predictions/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namePredictions/bias
q
$Predictions/bias/Read/ReadVariableOpReadVariableOpPredictions/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
åV
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* V
valueVBV BV
Ø
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

_init_input_shape* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
Õ
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*

.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 

4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
¦

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses*
Õ
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*

M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 

S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
¦

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
Õ
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses*

l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 

r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
¥
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|_random_generator
}__call__
*~&call_and_return_all_conditional_losses* 
­

kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	keras_api* 
* 

0
1
$2
%3
&4
'5
:6
;7
C8
D9
E10
F11
Y12
Z13
b14
c15
d16
e17
18
19*
k
0
1
$2
%3
:4
;5
C6
D7
Y8
Z9
b10
c11
12
13*


0* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
* 
]W
VARIABLE_VALUEConv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEConv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEBatchNorm_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEBatchNorm_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEBatchNorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEBatchNorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
$0
%1
&2
'3*

$0
%1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEConv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEConv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

:0
;1*

:0
;1*
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEBatchNorm_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEBatchNorm_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEBatchNorm_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEBatchNorm_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
C0
D1
E2
F3*

C0
D1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEConv_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEConv_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEBatchNorm_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEBatchNorm_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEBatchNorm_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEBatchNorm_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
b0
c1
d2
e3*

b0
c1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 
* 
* 
* 
b\
VARIABLE_VALUEPredictions/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEPredictions/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*


0* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
.
&0
'1
E2
F3
d4
e5*
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*

Õ0*
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

E0
F1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

d0
e1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


0* 
* 
<

Ötotal

×count
Ø	variables
Ù	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ö0
×1*

Ø	variables*

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
â
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Conv_1/kernelConv_1/biasBatchNorm_1/gammaBatchNorm_1/betaBatchNorm_1/moving_meanBatchNorm_1/moving_varianceConv_2/kernelConv_2/biasBatchNorm_2/gammaBatchNorm_2/betaBatchNorm_2/moving_meanBatchNorm_2/moving_varianceConv_3/kernelConv_3/biasBatchNorm_3/gammaBatchNorm_3/betaBatchNorm_3/moving_meanBatchNorm_3/moving_variancePredictions/kernelPredictions/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_5994
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Conv_1/kernel/Read/ReadVariableOpConv_1/bias/Read/ReadVariableOp%BatchNorm_1/gamma/Read/ReadVariableOp$BatchNorm_1/beta/Read/ReadVariableOp+BatchNorm_1/moving_mean/Read/ReadVariableOp/BatchNorm_1/moving_variance/Read/ReadVariableOp!Conv_2/kernel/Read/ReadVariableOpConv_2/bias/Read/ReadVariableOp%BatchNorm_2/gamma/Read/ReadVariableOp$BatchNorm_2/beta/Read/ReadVariableOp+BatchNorm_2/moving_mean/Read/ReadVariableOp/BatchNorm_2/moving_variance/Read/ReadVariableOp!Conv_3/kernel/Read/ReadVariableOpConv_3/bias/Read/ReadVariableOp%BatchNorm_3/gamma/Read/ReadVariableOp$BatchNorm_3/beta/Read/ReadVariableOp+BatchNorm_3/moving_mean/Read/ReadVariableOp/BatchNorm_3/moving_variance/Read/ReadVariableOp&Predictions/kernel/Read/ReadVariableOp$Predictions/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_6457
Æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv_1/kernelConv_1/biasBatchNorm_1/gammaBatchNorm_1/betaBatchNorm_1/moving_meanBatchNorm_1/moving_varianceConv_2/kernelConv_2/biasBatchNorm_2/gammaBatchNorm_2/betaBatchNorm_2/moving_meanBatchNorm_2/moving_varianceConv_3/kernelConv_3/biasBatchNorm_3/gammaBatchNorm_3/betaBatchNorm_3/moving_meanBatchNorm_3/moving_variancePredictions/kernelPredictions/biastotalcount*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_6533¿á

Î
´
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_6277

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
é

%__inference_Conv_1_layer_call_fn_6003

inputs!
unknown:		 
	unknown_0: 
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_1_layer_call_and_return_conditional_losses_5073w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
Å
*__inference_BatchNorm_1_layer_call_fn_6026

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4849
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
À

E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5001

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü
Å
*__inference_BatchNorm_3_layer_call_fn_6228

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5001
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«
D
(__inference_MaxPool_1_layer_call_fn_6090

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_4900
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
µ
__inference_loss_fn_0_6368O
=predictions_kernel_regularizer_square_readvariableop_resource:@
identity¢4Predictions/kernel/Regularizer/Square/ReadVariableOp²
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=predictions_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentity&Predictions/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: }
NoOpNoOp5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp
¤

ù
@__inference_Conv_2_layer_call_and_return_conditional_losses_6114

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï	
`
A__inference_Dropout_layer_call_and_return_conditional_losses_6325

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
àX
¸
 __inference__traced_restore_6533
file_prefix8
assignvariableop_conv_1_kernel:		 ,
assignvariableop_1_conv_1_bias: 2
$assignvariableop_2_batchnorm_1_gamma: 1
#assignvariableop_3_batchnorm_1_beta: 8
*assignvariableop_4_batchnorm_1_moving_mean: <
.assignvariableop_5_batchnorm_1_moving_variance: :
 assignvariableop_6_conv_2_kernel: @,
assignvariableop_7_conv_2_bias:@2
$assignvariableop_8_batchnorm_2_gamma:@1
#assignvariableop_9_batchnorm_2_beta:@9
+assignvariableop_10_batchnorm_2_moving_mean:@=
/assignvariableop_11_batchnorm_2_moving_variance:@;
!assignvariableop_12_conv_3_kernel:@@-
assignvariableop_13_conv_3_bias:@3
%assignvariableop_14_batchnorm_3_gamma:@2
$assignvariableop_15_batchnorm_3_beta:@9
+assignvariableop_16_batchnorm_3_moving_mean:@=
/assignvariableop_17_batchnorm_3_moving_variance:@8
&assignvariableop_18_predictions_kernel:@2
$assignvariableop_19_predictions_bias:#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ò

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_batchnorm_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_batchnorm_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_batchnorm_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batchnorm_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp$assignvariableop_8_batchnorm_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_batchnorm_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp+assignvariableop_10_batchnorm_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batchnorm_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp%assignvariableop_14_batchnorm_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_batchnorm_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp+assignvariableop_16_batchnorm_3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batchnorm_3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp&assignvariableop_18_predictions_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp$assignvariableop_19_predictions_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ä
\
@__inference_ReLu_2_layer_call_and_return_conditional_losses_5126

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ		@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs
À

E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6057

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
«
D
(__inference_MaxPool_2_layer_call_fn_6191

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_4976
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
ÿ
"__inference_signature_wrapper_5994
input_1!
unknown:		 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_4827k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ç

*__inference_Predictions_layer_call_fn_6340

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Predictions_layer_call_and_return_conditional_losses_5186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î
´
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6176

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò

(__inference_Night_Day_layer_call_fn_5530
input_1!
unknown:		 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Night_Day_layer_call_and_return_conditional_losses_5442k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ú
Å
*__inference_BatchNorm_1_layer_call_fn_6039

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4880
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨

ù
@__inference_Conv_1_layer_call_and_return_conditional_losses_6013

inputs8
conv2d_readvariableop_resource:		 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

ù
@__inference_Conv_3_layer_call_and_return_conditional_losses_6215

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ä
\
@__inference_ReLu_3_layer_call_and_return_conditional_losses_5159

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô
_
A__inference_Dropout_layer_call_and_return_conditional_losses_5167

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
K
À
C__inference_Night_Day_layer_call_and_return_conditional_losses_5442

inputs%
conv_1_5377:		 
conv_1_5379: 
batchnorm_1_5382: 
batchnorm_1_5384: 
batchnorm_1_5386: 
batchnorm_1_5388: %
conv_2_5393: @
conv_2_5395:@
batchnorm_2_5398:@
batchnorm_2_5400:@
batchnorm_2_5402:@
batchnorm_2_5404:@%
conv_3_5409:@@
conv_3_5411:@
batchnorm_3_5414:@
batchnorm_3_5416:@
batchnorm_3_5418:@
batchnorm_3_5420:@"
predictions_5426:@
predictions_5428:
identity¢#BatchNorm_1/StatefulPartitionedCall¢#BatchNorm_2/StatefulPartitionedCall¢#BatchNorm_3/StatefulPartitionedCall¢Conv_1/StatefulPartitionedCall¢Conv_2/StatefulPartitionedCall¢Conv_3/StatefulPartitionedCall¢Dropout/StatefulPartitionedCall¢#Predictions/StatefulPartitionedCall¢4Predictions/kernel/Regularizer/Square/ReadVariableOpí
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_5377conv_1_5379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_1_layer_call_and_return_conditional_losses_5073È
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_5382batchnorm_1_5384batchnorm_1_5386batchnorm_1_5388*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4880ã
ReLu_1/PartitionedCallPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_1_layer_call_and_return_conditional_losses_5093Ü
MaxPool_1/PartitionedCallPartitionedCallReLu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_4900
Conv_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv_2_5393conv_2_5395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_2_layer_call_and_return_conditional_losses_5106È
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall'Conv_2/StatefulPartitionedCall:output:0batchnorm_2_5398batchnorm_2_5400batchnorm_2_5402batchnorm_2_5404*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4956ã
ReLu_2/PartitionedCallPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_2_layer_call_and_return_conditional_losses_5126Ü
MaxPool_2/PartitionedCallPartitionedCallReLu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_4976
Conv_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv_3_5409conv_3_5411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_3_layer_call_and_return_conditional_losses_5139È
#BatchNorm_3/StatefulPartitionedCallStatefulPartitionedCall'Conv_3/StatefulPartitionedCall:output:0batchnorm_3_5414batchnorm_3_5416batchnorm_3_5418batchnorm_3_5420*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5032ã
ReLu_3/PartitionedCallPartitionedCall,BatchNorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_3_layer_call_and_return_conditional_losses_5159Ú
GlobaMaxPool/PartitionedCallPartitionedCallReLu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_5053æ
Dropout/StatefulPartitionedCallStatefulPartitionedCall%GlobaMaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Dropout_layer_call_and_return_conditional_losses_5276
#Predictions/StatefulPartitionedCallStatefulPartitionedCall(Dropout/StatefulPartitionedCall:output:0predictions_5426predictions_5428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Predictions_layer_call_and_return_conditional_losses_5186}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
&tf.__operators__.getitem/strided_sliceStridedSlice,Predictions/StatefulPartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOppredictions_5426*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity/tf.__operators__.getitem/strided_slice:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall$^BatchNorm_3/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall ^Dropout/StatefulPartitionedCall$^Predictions/StatefulPartitionedCall5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2J
#BatchNorm_3/StatefulPartitionedCall#BatchNorm_3/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2B
Dropout/StatefulPartitionedCallDropout/StatefulPartitionedCall2J
#Predictions/StatefulPartitionedCall#Predictions/StatefulPartitionedCall2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
­
E__inference_Predictions_layer_call_and_return_conditional_losses_6357

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢4Predictions/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

_
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_4976

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
Å
*__inference_BatchNorm_2_layer_call_fn_6127

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4925
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å

%__inference_Conv_2_layer_call_fn_6104

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_2_layer_call_and_return_conditional_losses_5106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤

ù
@__inference_Conv_3_layer_call_and_return_conditional_losses_5139

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü
G
+__inference_GlobaMaxPool_layer_call_fn_6292

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_5053i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
b
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_5053

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
A
%__inference_ReLu_3_layer_call_fn_6282

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_3_layer_call_and_return_conditional_losses_5159h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î
´
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4956

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ä
\
@__inference_ReLu_1_layer_call_and_return_conditional_losses_6085

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ?? :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? 
 
_user_specified_nameinputs
À

E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4849

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Õ

(__inference_Night_Day_layer_call_fn_5717

inputs!
unknown:		 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Night_Day_layer_call_and_return_conditional_losses_5203k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
\
@__inference_ReLu_3_layer_call_and_return_conditional_losses_6287

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ß
­
E__inference_Predictions_layer_call_and_return_conditional_losses_5186

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢4Predictions/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

_
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_6196

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
\
@__inference_ReLu_2_layer_call_and_return_conditional_losses_6186

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ		@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs
ì
_
&__inference_Dropout_layer_call_fn_6308

inputs
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Dropout_layer_call_and_return_conditional_losses_5276o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ú
Å
*__inference_BatchNorm_3_layer_call_fn_6241

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5032
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
él
×
__inference__wrapped_model_4827
input_1I
/night_day_conv_1_conv2d_readvariableop_resource:		 >
0night_day_conv_1_biasadd_readvariableop_resource: ;
-night_day_batchnorm_1_readvariableop_resource: =
/night_day_batchnorm_1_readvariableop_1_resource: L
>night_day_batchnorm_1_fusedbatchnormv3_readvariableop_resource: N
@night_day_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource: I
/night_day_conv_2_conv2d_readvariableop_resource: @>
0night_day_conv_2_biasadd_readvariableop_resource:@;
-night_day_batchnorm_2_readvariableop_resource:@=
/night_day_batchnorm_2_readvariableop_1_resource:@L
>night_day_batchnorm_2_fusedbatchnormv3_readvariableop_resource:@N
@night_day_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:@I
/night_day_conv_3_conv2d_readvariableop_resource:@@>
0night_day_conv_3_biasadd_readvariableop_resource:@;
-night_day_batchnorm_3_readvariableop_resource:@=
/night_day_batchnorm_3_readvariableop_1_resource:@L
>night_day_batchnorm_3_fusedbatchnormv3_readvariableop_resource:@N
@night_day_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource:@F
4night_day_predictions_matmul_readvariableop_resource:@C
5night_day_predictions_biasadd_readvariableop_resource:
identity¢5Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp¢7Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1¢$Night_Day/BatchNorm_1/ReadVariableOp¢&Night_Day/BatchNorm_1/ReadVariableOp_1¢5Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp¢7Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1¢$Night_Day/BatchNorm_2/ReadVariableOp¢&Night_Day/BatchNorm_2/ReadVariableOp_1¢5Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp¢7Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1¢$Night_Day/BatchNorm_3/ReadVariableOp¢&Night_Day/BatchNorm_3/ReadVariableOp_1¢'Night_Day/Conv_1/BiasAdd/ReadVariableOp¢&Night_Day/Conv_1/Conv2D/ReadVariableOp¢'Night_Day/Conv_2/BiasAdd/ReadVariableOp¢&Night_Day/Conv_2/Conv2D/ReadVariableOp¢'Night_Day/Conv_3/BiasAdd/ReadVariableOp¢&Night_Day/Conv_3/Conv2D/ReadVariableOp¢,Night_Day/Predictions/BiasAdd/ReadVariableOp¢+Night_Day/Predictions/MatMul/ReadVariableOp
&Night_Day/Conv_1/Conv2D/ReadVariableOpReadVariableOp/night_day_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0½
Night_Day/Conv_1/Conv2DConv2Dinput_1.Night_Day/Conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides

'Night_Day/Conv_1/BiasAdd/ReadVariableOpReadVariableOp0night_day_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
Night_Day/Conv_1/BiasAddBiasAdd Night_Day/Conv_1/Conv2D:output:0/Night_Day/Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? 
$Night_Day/BatchNorm_1/ReadVariableOpReadVariableOp-night_day_batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype0
&Night_Day/BatchNorm_1/ReadVariableOp_1ReadVariableOp/night_day_batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype0°
5Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>night_day_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0´
7Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@night_day_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¿
&Night_Day/BatchNorm_1/FusedBatchNormV3FusedBatchNormV3!Night_Day/Conv_1/BiasAdd:output:0,Night_Day/BatchNorm_1/ReadVariableOp:value:0.Night_Day/BatchNorm_1/ReadVariableOp_1:value:0=Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp:value:0?Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ?? : : : : :*
epsilon%o:*
is_training( 
Night_Day/ReLu_1/ReluRelu*Night_Day/BatchNorm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? ¸
Night_Day/MaxPool_1/MaxPoolMaxPool#Night_Day/ReLu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

&Night_Day/Conv_2/Conv2D/ReadVariableOpReadVariableOp/night_day_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ú
Night_Day/Conv_2/Conv2DConv2D$Night_Day/MaxPool_1/MaxPool:output:0.Night_Day/Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides

'Night_Day/Conv_2/BiasAdd/ReadVariableOpReadVariableOp0night_day_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0°
Night_Day/Conv_2/BiasAddBiasAdd Night_Day/Conv_2/Conv2D:output:0/Night_Day/Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
$Night_Day/BatchNorm_2/ReadVariableOpReadVariableOp-night_day_batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype0
&Night_Day/BatchNorm_2/ReadVariableOp_1ReadVariableOp/night_day_batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>night_day_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@night_day_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¿
&Night_Day/BatchNorm_2/FusedBatchNormV3FusedBatchNormV3!Night_Day/Conv_2/BiasAdd:output:0,Night_Day/BatchNorm_2/ReadVariableOp:value:0.Night_Day/BatchNorm_2/ReadVariableOp_1:value:0=Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp:value:0?Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ		@:@:@:@:@:*
epsilon%o:*
is_training( 
Night_Day/ReLu_2/ReluRelu*Night_Day/BatchNorm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@¸
Night_Day/MaxPool_2/MaxPoolMaxPool#Night_Day/ReLu_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

&Night_Day/Conv_3/Conv2D/ReadVariableOpReadVariableOp/night_day_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ú
Night_Day/Conv_3/Conv2DConv2D$Night_Day/MaxPool_2/MaxPool:output:0.Night_Day/Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

'Night_Day/Conv_3/BiasAdd/ReadVariableOpReadVariableOp0night_day_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0°
Night_Day/Conv_3/BiasAddBiasAdd Night_Day/Conv_3/Conv2D:output:0/Night_Day/Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$Night_Day/BatchNorm_3/ReadVariableOpReadVariableOp-night_day_batchnorm_3_readvariableop_resource*
_output_shapes
:@*
dtype0
&Night_Day/BatchNorm_3/ReadVariableOp_1ReadVariableOp/night_day_batchnorm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>night_day_batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@night_day_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0¿
&Night_Day/BatchNorm_3/FusedBatchNormV3FusedBatchNormV3!Night_Day/Conv_3/BiasAdd:output:0,Night_Day/BatchNorm_3/ReadVariableOp:value:0.Night_Day/BatchNorm_3/ReadVariableOp_1:value:0=Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp:value:0?Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
Night_Day/ReLu_3/ReluRelu*Night_Day/BatchNorm_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
,Night_Day/GlobaMaxPool/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ¯
Night_Day/GlobaMaxPool/MaxMax#Night_Day/ReLu_3/Relu:activations:05Night_Day/GlobaMaxPool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
Night_Day/Dropout/IdentityIdentity#Night_Day/GlobaMaxPool/Max:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
+Night_Day/Predictions/MatMul/ReadVariableOpReadVariableOp4night_day_predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0²
Night_Day/Predictions/MatMulMatMul#Night_Day/Dropout/Identity:output:03Night_Day/Predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,Night_Day/Predictions/BiasAdd/ReadVariableOpReadVariableOp5night_day_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
Night_Day/Predictions/BiasAddBiasAdd&Night_Day/Predictions/MatMul:product:04Night_Day/Predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Night_Day/Predictions/SoftmaxSoftmax&Night_Day/Predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6Night_Day/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
8Night_Day/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
8Night_Day/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¥
0Night_Day/tf.__operators__.getitem/strided_sliceStridedSlice'Night_Day/Predictions/Softmax:softmax:0?Night_Day/tf.__operators__.getitem/strided_slice/stack:output:0ANight_Day/tf.__operators__.getitem/strided_slice/stack_1:output:0ANight_Day/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
IdentityIdentity9Night_Day/tf.__operators__.getitem/strided_slice:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp6^Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp8^Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1%^Night_Day/BatchNorm_1/ReadVariableOp'^Night_Day/BatchNorm_1/ReadVariableOp_16^Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp8^Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1%^Night_Day/BatchNorm_2/ReadVariableOp'^Night_Day/BatchNorm_2/ReadVariableOp_16^Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp8^Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1%^Night_Day/BatchNorm_3/ReadVariableOp'^Night_Day/BatchNorm_3/ReadVariableOp_1(^Night_Day/Conv_1/BiasAdd/ReadVariableOp'^Night_Day/Conv_1/Conv2D/ReadVariableOp(^Night_Day/Conv_2/BiasAdd/ReadVariableOp'^Night_Day/Conv_2/Conv2D/ReadVariableOp(^Night_Day/Conv_3/BiasAdd/ReadVariableOp'^Night_Day/Conv_3/Conv2D/ReadVariableOp-^Night_Day/Predictions/BiasAdd/ReadVariableOp,^Night_Day/Predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2n
5Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp5Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp2r
7Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_17Night_Day/BatchNorm_1/FusedBatchNormV3/ReadVariableOp_12L
$Night_Day/BatchNorm_1/ReadVariableOp$Night_Day/BatchNorm_1/ReadVariableOp2P
&Night_Day/BatchNorm_1/ReadVariableOp_1&Night_Day/BatchNorm_1/ReadVariableOp_12n
5Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp5Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp2r
7Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_17Night_Day/BatchNorm_2/FusedBatchNormV3/ReadVariableOp_12L
$Night_Day/BatchNorm_2/ReadVariableOp$Night_Day/BatchNorm_2/ReadVariableOp2P
&Night_Day/BatchNorm_2/ReadVariableOp_1&Night_Day/BatchNorm_2/ReadVariableOp_12n
5Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp5Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp2r
7Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp_17Night_Day/BatchNorm_3/FusedBatchNormV3/ReadVariableOp_12L
$Night_Day/BatchNorm_3/ReadVariableOp$Night_Day/BatchNorm_3/ReadVariableOp2P
&Night_Day/BatchNorm_3/ReadVariableOp_1&Night_Day/BatchNorm_3/ReadVariableOp_12R
'Night_Day/Conv_1/BiasAdd/ReadVariableOp'Night_Day/Conv_1/BiasAdd/ReadVariableOp2P
&Night_Day/Conv_1/Conv2D/ReadVariableOp&Night_Day/Conv_1/Conv2D/ReadVariableOp2R
'Night_Day/Conv_2/BiasAdd/ReadVariableOp'Night_Day/Conv_2/BiasAdd/ReadVariableOp2P
&Night_Day/Conv_2/Conv2D/ReadVariableOp&Night_Day/Conv_2/Conv2D/ReadVariableOp2R
'Night_Day/Conv_3/BiasAdd/ReadVariableOp'Night_Day/Conv_3/BiasAdd/ReadVariableOp2P
&Night_Day/Conv_3/Conv2D/ReadVariableOp&Night_Day/Conv_3/Conv2D/ReadVariableOp2\
,Night_Day/Predictions/BiasAdd/ReadVariableOp,Night_Day/Predictions/BiasAdd/ReadVariableOp2Z
+Night_Day/Predictions/MatMul/ReadVariableOp+Night_Day/Predictions/MatMul/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ô
_
A__inference_Dropout_layer_call_and_return_conditional_losses_6313

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï

(__inference_Night_Day_layer_call_fn_5762

inputs!
unknown:		 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Night_Day_layer_call_and_return_conditional_losses_5442k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥e
¡
C__inference_Night_Day_layer_call_and_return_conditional_losses_5851

inputs?
%conv_1_conv2d_readvariableop_resource:		 4
&conv_1_biasadd_readvariableop_resource: 1
#batchnorm_1_readvariableop_resource: 3
%batchnorm_1_readvariableop_1_resource: B
4batchnorm_1_fusedbatchnormv3_readvariableop_resource: D
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_2_conv2d_readvariableop_resource: @4
&conv_2_biasadd_readvariableop_resource:@1
#batchnorm_2_readvariableop_resource:@3
%batchnorm_2_readvariableop_1_resource:@B
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:@D
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:@?
%conv_3_conv2d_readvariableop_resource:@@4
&conv_3_biasadd_readvariableop_resource:@1
#batchnorm_3_readvariableop_resource:@3
%batchnorm_3_readvariableop_1_resource:@B
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:@D
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource:@<
*predictions_matmul_readvariableop_resource:@9
+predictions_biasadd_readvariableop_resource:
identity¢+BatchNorm_1/FusedBatchNormV3/ReadVariableOp¢-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1¢BatchNorm_1/ReadVariableOp¢BatchNorm_1/ReadVariableOp_1¢+BatchNorm_2/FusedBatchNormV3/ReadVariableOp¢-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1¢BatchNorm_2/ReadVariableOp¢BatchNorm_2/ReadVariableOp_1¢+BatchNorm_3/FusedBatchNormV3/ReadVariableOp¢-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1¢BatchNorm_3/ReadVariableOp¢BatchNorm_3/ReadVariableOp_1¢Conv_1/BiasAdd/ReadVariableOp¢Conv_1/Conv2D/ReadVariableOp¢Conv_2/BiasAdd/ReadVariableOp¢Conv_2/Conv2D/ReadVariableOp¢Conv_3/BiasAdd/ReadVariableOp¢Conv_3/Conv2D/ReadVariableOp¢"Predictions/BiasAdd/ReadVariableOp¢!Predictions/MatMul/ReadVariableOp¢4Predictions/kernel/Regularizer/Square/ReadVariableOp
Conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0¨
Conv_1/Conv2DConv2Dinputs$Conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides

Conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
Conv_1/BiasAddBiasAddConv_1/Conv2D:output:0%Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? z
BatchNorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype0~
BatchNorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype0
+BatchNorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0 
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
BatchNorm_1/FusedBatchNormV3FusedBatchNormV3Conv_1/BiasAdd:output:0"BatchNorm_1/ReadVariableOp:value:0$BatchNorm_1/ReadVariableOp_1:value:03BatchNorm_1/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ?? : : : : :*
epsilon%o:*
is_training( o
ReLu_1/ReluRelu BatchNorm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? ¤
MaxPool_1/MaxPoolMaxPoolReLu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

Conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0¼
Conv_2/Conv2DConv2DMaxPool_1/MaxPool:output:0$Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides

Conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
Conv_2/BiasAddBiasAddConv_2/Conv2D:output:0%Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@z
BatchNorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype0~
BatchNorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0
+BatchNorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0 
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
BatchNorm_2/FusedBatchNormV3FusedBatchNormV3Conv_2/BiasAdd:output:0"BatchNorm_2/ReadVariableOp:value:0$BatchNorm_2/ReadVariableOp_1:value:03BatchNorm_2/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ		@:@:@:@:@:*
epsilon%o:*
is_training( o
ReLu_2/ReluRelu BatchNorm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@¤
MaxPool_2/MaxPoolMaxPoolReLu_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

Conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¼
Conv_3/Conv2DConv2DMaxPool_2/MaxPool:output:0$Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
Conv_3/BiasAddBiasAddConv_3/Conv2D:output:0%Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
BatchNorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes
:@*
dtype0~
BatchNorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
+BatchNorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0 
-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
BatchNorm_3/FusedBatchNormV3FusedBatchNormV3Conv_3/BiasAdd:output:0"BatchNorm_3/ReadVariableOp:value:0$BatchNorm_3/ReadVariableOp_1:value:03BatchNorm_3/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( o
ReLu_3/ReluRelu BatchNorm_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
"GlobaMaxPool/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
GlobaMaxPool/MaxMaxReLu_3/Relu:activations:0+GlobaMaxPool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
Dropout/IdentityIdentityGlobaMaxPool/Max:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!Predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
Predictions/MatMulMatMulDropout/Identity:output:0)Predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Predictions/BiasAddBiasAddPredictions/MatMul:product:0*Predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Predictions/SoftmaxSoftmaxPredictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ó
&tf.__operators__.getitem/strided_sliceStridedSlicePredictions/Softmax:softmax:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity/tf.__operators__.getitem/strided_slice:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp,^BatchNorm_1/FusedBatchNormV3/ReadVariableOp.^BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_1/ReadVariableOp^BatchNorm_1/ReadVariableOp_1,^BatchNorm_2/FusedBatchNormV3/ReadVariableOp.^BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_2/ReadVariableOp^BatchNorm_2/ReadVariableOp_1,^BatchNorm_3/FusedBatchNormV3/ReadVariableOp.^BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_3/ReadVariableOp^BatchNorm_3/ReadVariableOp_1^Conv_1/BiasAdd/ReadVariableOp^Conv_1/Conv2D/ReadVariableOp^Conv_2/BiasAdd/ReadVariableOp^Conv_2/Conv2D/ReadVariableOp^Conv_3/BiasAdd/ReadVariableOp^Conv_3/Conv2D/ReadVariableOp#^Predictions/BiasAdd/ReadVariableOp"^Predictions/MatMul/ReadVariableOp5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2Z
+BatchNorm_1/FusedBatchNormV3/ReadVariableOp+BatchNorm_1/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_1/ReadVariableOpBatchNorm_1/ReadVariableOp2<
BatchNorm_1/ReadVariableOp_1BatchNorm_1/ReadVariableOp_12Z
+BatchNorm_2/FusedBatchNormV3/ReadVariableOp+BatchNorm_2/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_2/ReadVariableOpBatchNorm_2/ReadVariableOp2<
BatchNorm_2/ReadVariableOp_1BatchNorm_2/ReadVariableOp_12Z
+BatchNorm_3/FusedBatchNormV3/ReadVariableOp+BatchNorm_3/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_3/ReadVariableOpBatchNorm_3/ReadVariableOp2<
BatchNorm_3/ReadVariableOp_1BatchNorm_3/ReadVariableOp_12>
Conv_1/BiasAdd/ReadVariableOpConv_1/BiasAdd/ReadVariableOp2<
Conv_1/Conv2D/ReadVariableOpConv_1/Conv2D/ReadVariableOp2>
Conv_2/BiasAdd/ReadVariableOpConv_2/BiasAdd/ReadVariableOp2<
Conv_2/Conv2D/ReadVariableOpConv_2/Conv2D/ReadVariableOp2>
Conv_3/BiasAdd/ReadVariableOpConv_3/BiasAdd/ReadVariableOp2<
Conv_3/Conv2D/ReadVariableOpConv_3/Conv2D/ReadVariableOp2H
"Predictions/BiasAdd/ReadVariableOp"Predictions/BiasAdd/ReadVariableOp2F
!Predictions/MatMul/ReadVariableOp!Predictions/MatMul/ReadVariableOp2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
Å
*__inference_BatchNorm_2_layer_call_fn_6140

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4956
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
À

E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6158

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å

%__inference_Conv_3_layer_call_fn_6205

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_3_layer_call_and_return_conditional_losses_5139w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
À

E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4925

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

_
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_6095

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´}
Õ
C__inference_Night_Day_layer_call_and_return_conditional_losses_5947

inputs?
%conv_1_conv2d_readvariableop_resource:		 4
&conv_1_biasadd_readvariableop_resource: 1
#batchnorm_1_readvariableop_resource: 3
%batchnorm_1_readvariableop_1_resource: B
4batchnorm_1_fusedbatchnormv3_readvariableop_resource: D
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_2_conv2d_readvariableop_resource: @4
&conv_2_biasadd_readvariableop_resource:@1
#batchnorm_2_readvariableop_resource:@3
%batchnorm_2_readvariableop_1_resource:@B
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:@D
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:@?
%conv_3_conv2d_readvariableop_resource:@@4
&conv_3_biasadd_readvariableop_resource:@1
#batchnorm_3_readvariableop_resource:@3
%batchnorm_3_readvariableop_1_resource:@B
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:@D
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource:@<
*predictions_matmul_readvariableop_resource:@9
+predictions_biasadd_readvariableop_resource:
identity¢BatchNorm_1/AssignNewValue¢BatchNorm_1/AssignNewValue_1¢+BatchNorm_1/FusedBatchNormV3/ReadVariableOp¢-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1¢BatchNorm_1/ReadVariableOp¢BatchNorm_1/ReadVariableOp_1¢BatchNorm_2/AssignNewValue¢BatchNorm_2/AssignNewValue_1¢+BatchNorm_2/FusedBatchNormV3/ReadVariableOp¢-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1¢BatchNorm_2/ReadVariableOp¢BatchNorm_2/ReadVariableOp_1¢BatchNorm_3/AssignNewValue¢BatchNorm_3/AssignNewValue_1¢+BatchNorm_3/FusedBatchNormV3/ReadVariableOp¢-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1¢BatchNorm_3/ReadVariableOp¢BatchNorm_3/ReadVariableOp_1¢Conv_1/BiasAdd/ReadVariableOp¢Conv_1/Conv2D/ReadVariableOp¢Conv_2/BiasAdd/ReadVariableOp¢Conv_2/Conv2D/ReadVariableOp¢Conv_3/BiasAdd/ReadVariableOp¢Conv_3/Conv2D/ReadVariableOp¢"Predictions/BiasAdd/ReadVariableOp¢!Predictions/MatMul/ReadVariableOp¢4Predictions/kernel/Regularizer/Square/ReadVariableOp
Conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0¨
Conv_1/Conv2DConv2Dinputs$Conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides

Conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
Conv_1/BiasAddBiasAddConv_1/Conv2D:output:0%Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? z
BatchNorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype0~
BatchNorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype0
+BatchNorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0 
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
BatchNorm_1/FusedBatchNormV3FusedBatchNormV3Conv_1/BiasAdd:output:0"BatchNorm_1/ReadVariableOp:value:0$BatchNorm_1/ReadVariableOp_1:value:03BatchNorm_1/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ?? : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<à
BatchNorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)BatchNorm_1/FusedBatchNormV3:batch_mean:0,^BatchNorm_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ê
BatchNorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-BatchNorm_1/FusedBatchNormV3:batch_variance:0.^BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0o
ReLu_1/ReluRelu BatchNorm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? ¤
MaxPool_1/MaxPoolMaxPoolReLu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

Conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0¼
Conv_2/Conv2DConv2DMaxPool_1/MaxPool:output:0$Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides

Conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
Conv_2/BiasAddBiasAddConv_2/Conv2D:output:0%Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@z
BatchNorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype0~
BatchNorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0
+BatchNorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0 
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
BatchNorm_2/FusedBatchNormV3FusedBatchNormV3Conv_2/BiasAdd:output:0"BatchNorm_2/ReadVariableOp:value:0$BatchNorm_2/ReadVariableOp_1:value:03BatchNorm_2/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ		@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<à
BatchNorm_2/AssignNewValueAssignVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource)BatchNorm_2/FusedBatchNormV3:batch_mean:0,^BatchNorm_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ê
BatchNorm_2/AssignNewValue_1AssignVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource-BatchNorm_2/FusedBatchNormV3:batch_variance:0.^BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0o
ReLu_2/ReluRelu BatchNorm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@¤
MaxPool_2/MaxPoolMaxPoolReLu_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

Conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¼
Conv_3/Conv2DConv2DMaxPool_2/MaxPool:output:0$Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
Conv_3/BiasAddBiasAddConv_3/Conv2D:output:0%Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
BatchNorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes
:@*
dtype0~
BatchNorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
+BatchNorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0 
-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
BatchNorm_3/FusedBatchNormV3FusedBatchNormV3Conv_3/BiasAdd:output:0"BatchNorm_3/ReadVariableOp:value:0$BatchNorm_3/ReadVariableOp_1:value:03BatchNorm_3/FusedBatchNormV3/ReadVariableOp:value:05BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<à
BatchNorm_3/AssignNewValueAssignVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource)BatchNorm_3/FusedBatchNormV3:batch_mean:0,^BatchNorm_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ê
BatchNorm_3/AssignNewValue_1AssignVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource-BatchNorm_3/FusedBatchNormV3:batch_variance:0.^BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0o
ReLu_3/ReluRelu BatchNorm_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
"GlobaMaxPool/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
GlobaMaxPool/MaxMaxReLu_3/Relu:activations:0+GlobaMaxPool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
Dropout/dropout/MulMulGlobaMaxPool/Max:output:0Dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
Dropout/dropout/ShapeShapeGlobaMaxPool/Max:output:0*
T0*
_output_shapes
:
,Dropout/dropout/random_uniform/RandomUniformRandomUniformDropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0c
Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¾
Dropout/dropout/GreaterEqualGreaterEqual5Dropout/dropout/random_uniform/RandomUniform:output:0'Dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dropout/dropout/CastCast Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Dropout/dropout/Mul_1MulDropout/dropout/Mul:z:0Dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!Predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
Predictions/MatMulMatMulDropout/dropout/Mul_1:z:0)Predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Predictions/BiasAddBiasAddPredictions/MatMul:product:0*Predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Predictions/SoftmaxSoftmaxPredictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ó
&tf.__operators__.getitem/strided_sliceStridedSlicePredictions/Softmax:softmax:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity/tf.__operators__.getitem/strided_slice:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BatchNorm_1/AssignNewValue^BatchNorm_1/AssignNewValue_1,^BatchNorm_1/FusedBatchNormV3/ReadVariableOp.^BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_1/ReadVariableOp^BatchNorm_1/ReadVariableOp_1^BatchNorm_2/AssignNewValue^BatchNorm_2/AssignNewValue_1,^BatchNorm_2/FusedBatchNormV3/ReadVariableOp.^BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_2/ReadVariableOp^BatchNorm_2/ReadVariableOp_1^BatchNorm_3/AssignNewValue^BatchNorm_3/AssignNewValue_1,^BatchNorm_3/FusedBatchNormV3/ReadVariableOp.^BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1^BatchNorm_3/ReadVariableOp^BatchNorm_3/ReadVariableOp_1^Conv_1/BiasAdd/ReadVariableOp^Conv_1/Conv2D/ReadVariableOp^Conv_2/BiasAdd/ReadVariableOp^Conv_2/Conv2D/ReadVariableOp^Conv_3/BiasAdd/ReadVariableOp^Conv_3/Conv2D/ReadVariableOp#^Predictions/BiasAdd/ReadVariableOp"^Predictions/MatMul/ReadVariableOp5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 28
BatchNorm_1/AssignNewValueBatchNorm_1/AssignNewValue2<
BatchNorm_1/AssignNewValue_1BatchNorm_1/AssignNewValue_12Z
+BatchNorm_1/FusedBatchNormV3/ReadVariableOp+BatchNorm_1/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_1/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_1/ReadVariableOpBatchNorm_1/ReadVariableOp2<
BatchNorm_1/ReadVariableOp_1BatchNorm_1/ReadVariableOp_128
BatchNorm_2/AssignNewValueBatchNorm_2/AssignNewValue2<
BatchNorm_2/AssignNewValue_1BatchNorm_2/AssignNewValue_12Z
+BatchNorm_2/FusedBatchNormV3/ReadVariableOp+BatchNorm_2/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_2/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_2/ReadVariableOpBatchNorm_2/ReadVariableOp2<
BatchNorm_2/ReadVariableOp_1BatchNorm_2/ReadVariableOp_128
BatchNorm_3/AssignNewValueBatchNorm_3/AssignNewValue2<
BatchNorm_3/AssignNewValue_1BatchNorm_3/AssignNewValue_12Z
+BatchNorm_3/FusedBatchNormV3/ReadVariableOp+BatchNorm_3/FusedBatchNormV3/ReadVariableOp2^
-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_1-BatchNorm_3/FusedBatchNormV3/ReadVariableOp_128
BatchNorm_3/ReadVariableOpBatchNorm_3/ReadVariableOp2<
BatchNorm_3/ReadVariableOp_1BatchNorm_3/ReadVariableOp_12>
Conv_1/BiasAdd/ReadVariableOpConv_1/BiasAdd/ReadVariableOp2<
Conv_1/Conv2D/ReadVariableOpConv_1/Conv2D/ReadVariableOp2>
Conv_2/BiasAdd/ReadVariableOpConv_2/BiasAdd/ReadVariableOp2<
Conv_2/Conv2D/ReadVariableOpConv_2/Conv2D/ReadVariableOp2>
Conv_3/BiasAdd/ReadVariableOpConv_3/BiasAdd/ReadVariableOp2<
Conv_3/Conv2D/ReadVariableOpConv_3/Conv2D/ReadVariableOp2H
"Predictions/BiasAdd/ReadVariableOp"Predictions/BiasAdd/ReadVariableOp2F
!Predictions/MatMul/ReadVariableOp!Predictions/MatMul/ReadVariableOp2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
A
%__inference_ReLu_2_layer_call_fn_6181

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_2_layer_call_and_return_conditional_losses_5126h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ		@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs
K
Á
C__inference_Night_Day_layer_call_and_return_conditional_losses_5666
input_1%
conv_1_5601:		 
conv_1_5603: 
batchnorm_1_5606: 
batchnorm_1_5608: 
batchnorm_1_5610: 
batchnorm_1_5612: %
conv_2_5617: @
conv_2_5619:@
batchnorm_2_5622:@
batchnorm_2_5624:@
batchnorm_2_5626:@
batchnorm_2_5628:@%
conv_3_5633:@@
conv_3_5635:@
batchnorm_3_5638:@
batchnorm_3_5640:@
batchnorm_3_5642:@
batchnorm_3_5644:@"
predictions_5650:@
predictions_5652:
identity¢#BatchNorm_1/StatefulPartitionedCall¢#BatchNorm_2/StatefulPartitionedCall¢#BatchNorm_3/StatefulPartitionedCall¢Conv_1/StatefulPartitionedCall¢Conv_2/StatefulPartitionedCall¢Conv_3/StatefulPartitionedCall¢Dropout/StatefulPartitionedCall¢#Predictions/StatefulPartitionedCall¢4Predictions/kernel/Regularizer/Square/ReadVariableOpî
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_1_5601conv_1_5603*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_1_layer_call_and_return_conditional_losses_5073È
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_5606batchnorm_1_5608batchnorm_1_5610batchnorm_1_5612*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4880ã
ReLu_1/PartitionedCallPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_1_layer_call_and_return_conditional_losses_5093Ü
MaxPool_1/PartitionedCallPartitionedCallReLu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_4900
Conv_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv_2_5617conv_2_5619*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_2_layer_call_and_return_conditional_losses_5106È
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall'Conv_2/StatefulPartitionedCall:output:0batchnorm_2_5622batchnorm_2_5624batchnorm_2_5626batchnorm_2_5628*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4956ã
ReLu_2/PartitionedCallPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_2_layer_call_and_return_conditional_losses_5126Ü
MaxPool_2/PartitionedCallPartitionedCallReLu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_4976
Conv_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv_3_5633conv_3_5635*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_3_layer_call_and_return_conditional_losses_5139È
#BatchNorm_3/StatefulPartitionedCallStatefulPartitionedCall'Conv_3/StatefulPartitionedCall:output:0batchnorm_3_5638batchnorm_3_5640batchnorm_3_5642batchnorm_3_5644*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5032ã
ReLu_3/PartitionedCallPartitionedCall,BatchNorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_3_layer_call_and_return_conditional_losses_5159Ú
GlobaMaxPool/PartitionedCallPartitionedCallReLu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_5053æ
Dropout/StatefulPartitionedCallStatefulPartitionedCall%GlobaMaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Dropout_layer_call_and_return_conditional_losses_5276
#Predictions/StatefulPartitionedCallStatefulPartitionedCall(Dropout/StatefulPartitionedCall:output:0predictions_5650predictions_5652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Predictions_layer_call_and_return_conditional_losses_5186}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
&tf.__operators__.getitem/strided_sliceStridedSlice,Predictions/StatefulPartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOppredictions_5650*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity/tf.__operators__.getitem/strided_slice:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall$^BatchNorm_3/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall ^Dropout/StatefulPartitionedCall$^Predictions/StatefulPartitionedCall5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2J
#BatchNorm_3/StatefulPartitionedCall#BatchNorm_3/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2B
Dropout/StatefulPartitionedCallDropout/StatefulPartitionedCall2J
#Predictions/StatefulPartitionedCall#Predictions/StatefulPartitionedCall2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
À

E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_6259

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø

(__inference_Night_Day_layer_call_fn_5246
input_1!
unknown:		 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Night_Day_layer_call_and_return_conditional_losses_5203k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ä
\
@__inference_ReLu_1_layer_call_and_return_conditional_losses_5093

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ?? :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? 
 
_user_specified_nameinputs
¨

ù
@__inference_Conv_1_layer_call_and_return_conditional_losses_5073

inputs8
conv2d_readvariableop_resource:		 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_4900

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
´
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5032

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

ù
@__inference_Conv_2_layer_call_and_return_conditional_losses_5106

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

B
&__inference_Dropout_layer_call_fn_6303

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Dropout_layer_call_and_return_conditional_losses_5167`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î
´
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6075

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
þI

C__inference_Night_Day_layer_call_and_return_conditional_losses_5203

inputs%
conv_1_5074:		 
conv_1_5076: 
batchnorm_1_5079: 
batchnorm_1_5081: 
batchnorm_1_5083: 
batchnorm_1_5085: %
conv_2_5107: @
conv_2_5109:@
batchnorm_2_5112:@
batchnorm_2_5114:@
batchnorm_2_5116:@
batchnorm_2_5118:@%
conv_3_5140:@@
conv_3_5142:@
batchnorm_3_5145:@
batchnorm_3_5147:@
batchnorm_3_5149:@
batchnorm_3_5151:@"
predictions_5187:@
predictions_5189:
identity¢#BatchNorm_1/StatefulPartitionedCall¢#BatchNorm_2/StatefulPartitionedCall¢#BatchNorm_3/StatefulPartitionedCall¢Conv_1/StatefulPartitionedCall¢Conv_2/StatefulPartitionedCall¢Conv_3/StatefulPartitionedCall¢#Predictions/StatefulPartitionedCall¢4Predictions/kernel/Regularizer/Square/ReadVariableOpí
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_5074conv_1_5076*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_1_layer_call_and_return_conditional_losses_5073Ê
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_5079batchnorm_1_5081batchnorm_1_5083batchnorm_1_5085*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4849ã
ReLu_1/PartitionedCallPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_1_layer_call_and_return_conditional_losses_5093Ü
MaxPool_1/PartitionedCallPartitionedCallReLu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_4900
Conv_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv_2_5107conv_2_5109*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_2_layer_call_and_return_conditional_losses_5106Ê
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall'Conv_2/StatefulPartitionedCall:output:0batchnorm_2_5112batchnorm_2_5114batchnorm_2_5116batchnorm_2_5118*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4925ã
ReLu_2/PartitionedCallPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_2_layer_call_and_return_conditional_losses_5126Ü
MaxPool_2/PartitionedCallPartitionedCallReLu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_4976
Conv_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv_3_5140conv_3_5142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_3_layer_call_and_return_conditional_losses_5139Ê
#BatchNorm_3/StatefulPartitionedCallStatefulPartitionedCall'Conv_3/StatefulPartitionedCall:output:0batchnorm_3_5145batchnorm_3_5147batchnorm_3_5149batchnorm_3_5151*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5001ã
ReLu_3/PartitionedCallPartitionedCall,BatchNorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_3_layer_call_and_return_conditional_losses_5159Ú
GlobaMaxPool/PartitionedCallPartitionedCallReLu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_5053Ö
Dropout/PartitionedCallPartitionedCall%GlobaMaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Dropout_layer_call_and_return_conditional_losses_5167
#Predictions/StatefulPartitionedCallStatefulPartitionedCall Dropout/PartitionedCall:output:0predictions_5187predictions_5189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Predictions_layer_call_and_return_conditional_losses_5186}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
&tf.__operators__.getitem/strided_sliceStridedSlice,Predictions/StatefulPartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOppredictions_5187*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity/tf.__operators__.getitem/strided_slice:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall$^BatchNorm_3/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall$^Predictions/StatefulPartitionedCall5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2J
#BatchNorm_3/StatefulPartitionedCall#BatchNorm_3/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2J
#Predictions/StatefulPartitionedCall#Predictions/StatefulPartitionedCall2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï	
`
A__inference_Dropout_layer_call_and_return_conditional_losses_5276

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
3
­	
__inference__traced_save_6457
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop0
,savev2_batchnorm_1_gamma_read_readvariableop/
+savev2_batchnorm_1_beta_read_readvariableop6
2savev2_batchnorm_1_moving_mean_read_readvariableop:
6savev2_batchnorm_1_moving_variance_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop0
,savev2_batchnorm_2_gamma_read_readvariableop/
+savev2_batchnorm_2_beta_read_readvariableop6
2savev2_batchnorm_2_moving_mean_read_readvariableop:
6savev2_batchnorm_2_moving_variance_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop0
,savev2_batchnorm_3_gamma_read_readvariableop/
+savev2_batchnorm_3_beta_read_readvariableop6
2savev2_batchnorm_3_moving_mean_read_readvariableop:
6savev2_batchnorm_3_moving_variance_read_readvariableop1
-savev2_predictions_kernel_read_readvariableop/
+savev2_predictions_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ï

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ±	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop,savev2_batchnorm_1_gamma_read_readvariableop+savev2_batchnorm_1_beta_read_readvariableop2savev2_batchnorm_1_moving_mean_read_readvariableop6savev2_batchnorm_1_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop,savev2_batchnorm_2_gamma_read_readvariableop+savev2_batchnorm_2_beta_read_readvariableop2savev2_batchnorm_2_moving_mean_read_readvariableop6savev2_batchnorm_2_moving_variance_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop,savev2_batchnorm_3_gamma_read_readvariableop+savev2_batchnorm_3_beta_read_readvariableop2savev2_batchnorm_3_moving_mean_read_readvariableop6savev2_batchnorm_3_moving_variance_read_readvariableop-savev2_predictions_kernel_read_readvariableop+savev2_predictions_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*½
_input_shapes«
¨: :		 : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		 : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¸
A
%__inference_ReLu_1_layer_call_fn_6080

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_1_layer_call_and_return_conditional_losses_5093h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ?? :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? 
 
_user_specified_nameinputs
Î
´
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4880

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0º
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢
b
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_6298

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
J

C__inference_Night_Day_layer_call_and_return_conditional_losses_5598
input_1%
conv_1_5533:		 
conv_1_5535: 
batchnorm_1_5538: 
batchnorm_1_5540: 
batchnorm_1_5542: 
batchnorm_1_5544: %
conv_2_5549: @
conv_2_5551:@
batchnorm_2_5554:@
batchnorm_2_5556:@
batchnorm_2_5558:@
batchnorm_2_5560:@%
conv_3_5565:@@
conv_3_5567:@
batchnorm_3_5570:@
batchnorm_3_5572:@
batchnorm_3_5574:@
batchnorm_3_5576:@"
predictions_5582:@
predictions_5584:
identity¢#BatchNorm_1/StatefulPartitionedCall¢#BatchNorm_2/StatefulPartitionedCall¢#BatchNorm_3/StatefulPartitionedCall¢Conv_1/StatefulPartitionedCall¢Conv_2/StatefulPartitionedCall¢Conv_3/StatefulPartitionedCall¢#Predictions/StatefulPartitionedCall¢4Predictions/kernel/Regularizer/Square/ReadVariableOpî
Conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_1_5533conv_1_5535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_1_layer_call_and_return_conditional_losses_5073Ê
#BatchNorm_1/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0batchnorm_1_5538batchnorm_1_5540batchnorm_1_5542batchnorm_1_5544*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_4849ã
ReLu_1/PartitionedCallPartitionedCall,BatchNorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_1_layer_call_and_return_conditional_losses_5093Ü
MaxPool_1/PartitionedCallPartitionedCallReLu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_4900
Conv_2/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_1/PartitionedCall:output:0conv_2_5549conv_2_5551*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_2_layer_call_and_return_conditional_losses_5106Ê
#BatchNorm_2/StatefulPartitionedCallStatefulPartitionedCall'Conv_2/StatefulPartitionedCall:output:0batchnorm_2_5554batchnorm_2_5556batchnorm_2_5558batchnorm_2_5560*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_4925ã
ReLu_2/PartitionedCallPartitionedCall,BatchNorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_2_layer_call_and_return_conditional_losses_5126Ü
MaxPool_2/PartitionedCallPartitionedCallReLu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_4976
Conv_3/StatefulPartitionedCallStatefulPartitionedCall"MaxPool_2/PartitionedCall:output:0conv_3_5565conv_3_5567*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Conv_3_layer_call_and_return_conditional_losses_5139Ê
#BatchNorm_3/StatefulPartitionedCallStatefulPartitionedCall'Conv_3/StatefulPartitionedCall:output:0batchnorm_3_5570batchnorm_3_5572batchnorm_3_5574batchnorm_3_5576*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_5001ã
ReLu_3/PartitionedCallPartitionedCall,BatchNorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ReLu_3_layer_call_and_return_conditional_losses_5159Ú
GlobaMaxPool/PartitionedCallPartitionedCallReLu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_5053Ö
Dropout/PartitionedCallPartitionedCall%GlobaMaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Dropout_layer_call_and_return_conditional_losses_5167
#Predictions/StatefulPartitionedCallStatefulPartitionedCall Dropout/PartitionedCall:output:0predictions_5582predictions_5584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Predictions_layer_call_and_return_conditional_losses_5186}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
&tf.__operators__.getitem/strided_sliceStridedSlice,Predictions/StatefulPartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask
4Predictions/kernel/Regularizer/Square/ReadVariableOpReadVariableOppredictions_5582*
_output_shapes

:@*
dtype0
%Predictions/kernel/Regularizer/SquareSquare<Predictions/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@u
$Predictions/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"Predictions/kernel/Regularizer/SumSum)Predictions/kernel/Regularizer/Square:y:0-Predictions/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: i
$Predictions/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"Predictions/kernel/Regularizer/mulMul-Predictions/kernel/Regularizer/mul/x:output:0+Predictions/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity/tf.__operators__.getitem/strided_slice:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
NoOpNoOp$^BatchNorm_1/StatefulPartitionedCall$^BatchNorm_2/StatefulPartitionedCall$^BatchNorm_3/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall$^Predictions/StatefulPartitionedCall5^Predictions/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2J
#BatchNorm_1/StatefulPartitionedCall#BatchNorm_1/StatefulPartitionedCall2J
#BatchNorm_2/StatefulPartitionedCall#BatchNorm_2/StatefulPartitionedCall2J
#BatchNorm_3/StatefulPartitionedCall#BatchNorm_3/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2J
#Predictions/StatefulPartitionedCall#Predictions/StatefulPartitionedCall2l
4Predictions/kernel/Regularizer/Square/ReadVariableOp4Predictions/kernel/Regularizer/Square/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Á
serving_default­
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿH
tf.__operators__.getitem,
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:úð
ï
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
»

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|_random_generator
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
Â

kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
"
	optimizer
·
0
1
$2
%3
&4
'5
:6
;7
C8
D9
E10
F11
Y12
Z13
b14
c15
d16
e17
18
19"
trackable_list_wrapper

0
1
$2
%3
:4
;5
C6
D7
Y8
Z9
b10
c11
12
13"
trackable_list_wrapper
(
0"
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î2ë
(__inference_Night_Day_layer_call_fn_5246
(__inference_Night_Day_layer_call_fn_5717
(__inference_Night_Day_layer_call_fn_5762
(__inference_Night_Day_layer_call_fn_5530À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_Night_Day_layer_call_and_return_conditional_losses_5851
C__inference_Night_Day_layer_call_and_return_conditional_losses_5947
C__inference_Night_Day_layer_call_and_return_conditional_losses_5598
C__inference_Night_Day_layer_call_and_return_conditional_losses_5666À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÊBÇ
__inference__wrapped_model_4827input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
serving_default"
signature_map
 "
trackable_list_wrapper
':%		 2Conv_1/kernel
: 2Conv_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_Conv_1_layer_call_fn_6003¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_Conv_1_layer_call_and_return_conditional_losses_6013¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
: 2BatchNorm_1/gamma
: 2BatchNorm_1/beta
':%  (2BatchNorm_1/moving_mean
+:)  (2BatchNorm_1/moving_variance
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
2
*__inference_BatchNorm_1_layer_call_fn_6026
*__inference_BatchNorm_1_layer_call_fn_6039´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6057
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6075´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_ReLu_1_layer_call_fn_6080¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_ReLu_1_layer_call_and_return_conditional_losses_6085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_MaxPool_1_layer_call_fn_6090¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_6095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':% @2Conv_2/kernel
:@2Conv_2/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_Conv_2_layer_call_fn_6104¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_Conv_2_layer_call_and_return_conditional_losses_6114¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
:@2BatchNorm_2/gamma
:@2BatchNorm_2/beta
':%@ (2BatchNorm_2/moving_mean
+:)@ (2BatchNorm_2/moving_variance
<
C0
D1
E2
F3"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
2
*__inference_BatchNorm_2_layer_call_fn_6127
*__inference_BatchNorm_2_layer_call_fn_6140´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6158
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6176´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_ReLu_2_layer_call_fn_6181¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_ReLu_2_layer_call_and_return_conditional_losses_6186¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_MaxPool_2_layer_call_fn_6191¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_6196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%@@2Conv_3/kernel
:@2Conv_3/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_Conv_3_layer_call_fn_6205¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_Conv_3_layer_call_and_return_conditional_losses_6215¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
:@2BatchNorm_3/gamma
:@2BatchNorm_3/beta
':%@ (2BatchNorm_3/moving_mean
+:)@ (2BatchNorm_3/moving_variance
<
b0
c1
d2
e3"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
2
*__inference_BatchNorm_3_layer_call_fn_6228
*__inference_BatchNorm_3_layer_call_fn_6241´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_6259
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_6277´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_ReLu_3_layer_call_fn_6282¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_ReLu_3_layer_call_and_return_conditional_losses_6287¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_GlobaMaxPool_layer_call_fn_6292¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_6298¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
&__inference_Dropout_layer_call_fn_6303
&__inference_Dropout_layer_call_fn_6308´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
A__inference_Dropout_layer_call_and_return_conditional_losses_6313
A__inference_Dropout_layer_call_and_return_conditional_losses_6325´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
$:"@2Predictions/kernel
:2Predictions/bias
/
0
1"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_Predictions_layer_call_fn_6340¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_Predictions_layer_call_and_return_conditional_losses_6357¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
±2®
__inference_loss_fn_0_6368
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
J
&0
'1
E2
F3
d4
e5"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
(
Õ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÉBÆ
"__inference_signature_wrapper_5994input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Ötotal

×count
Ø	variables
Ù	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
Ö0
×1"
trackable_list_wrapper
.
Ø	variables"
_generic_user_objectà
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6057$%&'M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 à
E__inference_BatchNorm_1_layer_call_and_return_conditional_losses_6075$%&'M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¸
*__inference_BatchNorm_1_layer_call_fn_6026$%&'M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¸
*__inference_BatchNorm_1_layer_call_fn_6039$%&'M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ à
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6158CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 à
E__inference_BatchNorm_2_layer_call_and_return_conditional_losses_6176CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¸
*__inference_BatchNorm_2_layer_call_fn_6127CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
*__inference_BatchNorm_2_layer_call_fn_6140CDEFM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@à
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_6259bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 à
E__inference_BatchNorm_3_layer_call_and_return_conditional_losses_6277bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¸
*__inference_BatchNorm_3_layer_call_fn_6228bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
*__inference_BatchNorm_3_layer_call_fn_6241bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@²
@__inference_Conv_1_layer_call_and_return_conditional_losses_6013n9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ?? 
 
%__inference_Conv_1_layer_call_fn_6003a9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ?? °
@__inference_Conv_2_layer_call_and_return_conditional_losses_6114l:;7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ		@
 
%__inference_Conv_2_layer_call_fn_6104_:;7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ		@°
@__inference_Conv_3_layer_call_and_return_conditional_losses_6215lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
%__inference_Conv_3_layer_call_fn_6205_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¡
A__inference_Dropout_layer_call_and_return_conditional_losses_6313\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¡
A__inference_Dropout_layer_call_and_return_conditional_losses_6325\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 y
&__inference_Dropout_layer_call_fn_6303O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@y
&__inference_Dropout_layer_call_fn_6308O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@Ï
F__inference_GlobaMaxPool_layer_call_and_return_conditional_losses_6298R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¦
+__inference_GlobaMaxPool_layer_call_fn_6292wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
C__inference_MaxPool_1_layer_call_and_return_conditional_losses_6095R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
(__inference_MaxPool_1_layer_call_fn_6090R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
C__inference_MaxPool_2_layer_call_and_return_conditional_losses_6196R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
(__inference_MaxPool_2_layer_call_fn_6191R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
C__inference_Night_Day_layer_call_and_return_conditional_losses_5598~$%&':;CDEFYZbcdeB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "!¢

0ÿÿÿÿÿÿÿÿÿ
 Å
C__inference_Night_Day_layer_call_and_return_conditional_losses_5666~$%&':;CDEFYZbcdeB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "!¢

0ÿÿÿÿÿÿÿÿÿ
 Ä
C__inference_Night_Day_layer_call_and_return_conditional_losses_5851}$%&':;CDEFYZbcdeA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "!¢

0ÿÿÿÿÿÿÿÿÿ
 Ä
C__inference_Night_Day_layer_call_and_return_conditional_losses_5947}$%&':;CDEFYZbcdeA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "!¢

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_Night_Day_layer_call_fn_5246q$%&':;CDEFYZbcdeB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_Night_Day_layer_call_fn_5530q$%&':;CDEFYZbcdeB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_Night_Day_layer_call_fn_5717p$%&':;CDEFYZbcdeA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_Night_Day_layer_call_fn_5762p$%&':;CDEFYZbcdeA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_Predictions_layer_call_and_return_conditional_losses_6357]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_Predictions_layer_call_fn_6340P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¬
@__inference_ReLu_1_layer_call_and_return_conditional_losses_6085h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ?? 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ?? 
 
%__inference_ReLu_1_layer_call_fn_6080[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ?? 
ª " ÿÿÿÿÿÿÿÿÿ?? ¬
@__inference_ReLu_2_layer_call_and_return_conditional_losses_6186h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ		@
 
%__inference_ReLu_2_layer_call_fn_6181[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª " ÿÿÿÿÿÿÿÿÿ		@¬
@__inference_ReLu_3_layer_call_and_return_conditional_losses_6287h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
%__inference_ReLu_3_layer_call_fn_6282[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@È
__inference__wrapped_model_4827¤$%&':;CDEFYZbcde:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ
ª "OªL
J
tf.__operators__.getitem.+
tf.__operators__.getitemÿÿÿÿÿÿÿÿÿ9
__inference_loss_fn_0_6368¢

¢ 
ª " Ö
"__inference_signature_wrapper_5994¯$%&':;CDEFYZbcdeE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿ"OªL
J
tf.__operators__.getitem.+
tf.__operators__.getitemÿÿÿÿÿÿÿÿÿ