é!
¶
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
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758Óë
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:		 *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
r
pred/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namepred/kernel
k
pred/kernel/Read/ReadVariableOpReadVariableOppred/kernel*
_output_shapes

:@*
dtype0
j
	pred/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	pred/bias
c
pred/bias/Read/ReadVariableOpReadVariableOp	pred/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

random_rotation_1/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*+
shared_namerandom_rotation_1/StateVar

.random_rotation_1/StateVar/Read/ReadVariableOpReadVariableOprandom_rotation_1/StateVar*
_output_shapes
:*
dtype0	

random_translation_1/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*.
shared_namerandom_translation_1/StateVar

1random_translation_1/StateVar/Read/ReadVariableOpReadVariableOprandom_translation_1/StateVar*
_output_shapes
:*
dtype0	

random_flip_1/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*'
shared_namerandom_flip_1/StateVar
}
*random_flip_1/StateVar/Read/ReadVariableOpReadVariableOprandom_flip_1/StateVar*
_output_shapes
:*
dtype0	

random_contrast_1/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*+
shared_namerandom_contrast_1/StateVar

.random_contrast_1/StateVar/Read/ReadVariableOpReadVariableOprandom_contrast_1/StateVar*
_output_shapes
:*
dtype0	

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:		 *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
: *
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0

Adam/pred/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_nameAdam/pred/kernel/m
y
&Adam/pred/kernel/m/Read/ReadVariableOpReadVariableOpAdam/pred/kernel/m*
_output_shapes

:@*
dtype0
x
Adam/pred/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/pred/bias/m
q
$Adam/pred/bias/m/Read/ReadVariableOpReadVariableOpAdam/pred/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:		 *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
: *
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0

Adam/pred/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_nameAdam/pred/kernel/v
y
&Adam/pred/kernel/v/Read/ReadVariableOpReadVariableOpAdam/pred/kernel/v*
_output_shapes

:@*
dtype0
x
Adam/pred/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/pred/bias/v
q
$Adam/pred/bias/v/Read/ReadVariableOpReadVariableOpAdam/pred/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¹
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ó
valueèBä BÜ
Ø
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
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
Ä
layer-0
layer-1
layer-2
layer-3
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
¦

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
Õ
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 

>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
¦

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
Õ
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses*

W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses* 

]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
¦

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses*
Õ
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses*

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 

|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
å
	iter
beta_1
beta_2

decay
learning_rate%m«&m¬.m­/m®Dm¯Em°Mm±Nm²cm³dm´lmµmm¶	m·	m¸%v¹&vº.v»/v¼Dv½Ev¾Mv¿NvÀcvÁdvÂlvÃmvÄ	vÅ	vÆ*

%0
&1
.2
/3
04
15
D6
E7
M8
N9
O10
P11
c12
d13
l14
m15
n16
o17
18
19*
l
%0
&1
.2
/3
D4
E5
M6
N7
c8
d9
l10
m11
12
13*


0* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
serving_default* 
* 
®
	variables
trainable_variables
regularization_losses
 	keras_api
¡_random_generator
¢__call__
+£&call_and_return_all_conditional_losses*
®
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨_random_generator
©__call__
+ª&call_and_return_all_conditional_losses*
®
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯_random_generator
°__call__
+±&call_and_return_all_conditional_losses*
®
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶_random_generator
·__call__
+¸&call_and_return_all_conditional_losses*
* 
* 
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 

¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
.0
/1
02
13*

.0
/1*
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
M0
N1
O2
P3*

M0
N1*
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

c0
d1*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
l0
m1
n2
o3*

l0
m1*
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
[U
VARIABLE_VALUEpred/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	pred/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*


0* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
00
11
O2
P3
n4
o5*
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

0
1*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses* 


_generator*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses* 


_generator*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
«	variables
¬trainable_variables
­regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses* 


_generator*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses* 


_generator*
* 
* 
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 

00
11*
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
O0
P1*
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
n0
o1*
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


0* 
* 
<

total

count
 	variables
¡	keras_api*
M

¢total

£count
¤
_fn_kwargs
¥	variables
¦	keras_api*
* 
* 
* 
* 
* 

§
_state_var*
* 
* 
* 
* 
* 

¨
_state_var*
* 
* 
* 
* 
* 

©
_state_var*
* 
* 
* 
* 
* 

ª
_state_var*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

 	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¢0
£1*

¥	variables*

VARIABLE_VALUErandom_rotation_1/StateVarRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUErandom_translation_1/StateVarRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUErandom_flip_1/StateVarRlayer-1/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUErandom_contrast_1/StateVarRlayer-1/layer-3/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/pred/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/pred/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/pred/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/pred/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
Ñ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancepred/kernel	pred/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_40015
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
á
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOppred/kernel/Read/ReadVariableOppred/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.random_rotation_1/StateVar/Read/ReadVariableOp1random_translation_1/StateVar/Read/ReadVariableOp*random_flip_1/StateVar/Read/ReadVariableOp.random_contrast_1/StateVar/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp&Adam/pred/kernel/m/Read/ReadVariableOp$Adam/pred/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp&Adam/pred/kernel/v/Read/ReadVariableOp$Adam/pred/bias/v/Read/ReadVariableOpConst*J
TinC
A2?					*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_41355

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancepred/kernel	pred/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1random_rotation_1/StateVarrandom_translation_1/StateVarrandom_flip_1/StateVarrandom_contrast_1/StateVarAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/pred/kernel/mAdam/pred/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/pred/kernel/vAdam/pred/bias/v*I
TinB
@2>*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_41548ø¦
ö
d
+__inference_top_dropout_layer_call_fn_40672

inputs
identity¢StatefulPartitionedCallÄ
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
GPU2*0J 8 *O
fJRH
F__inference_top_dropout_layer_call_and_return_conditional_losses_38951o
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
Ù
¿
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38710

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
½
x
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38489
random_rotation_1_input
identityç
!random_rotation_1/PartitionedCallPartitionedCallrandom_rotation_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38025
$random_translation_1/PartitionedCallPartitionedCall*random_rotation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38031õ
random_flip_1/PartitionedCallPartitionedCall-random_translation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38037ö
!random_contrast_1/PartitionedCallPartitionedCall&random_flip_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38043|
IdentityIdentity*random_contrast_1/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:j f
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1
_user_specified_namerandom_rotation_1_input
áI
²	
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_38878

inputs&
conv2d_38753:		 
conv2d_38755: '
batch_normalization_38758: '
batch_normalization_38760: '
batch_normalization_38762: '
batch_normalization_38764: (
conv2d_1_38786: @
conv2d_1_38788:@)
batch_normalization_1_38791:@)
batch_normalization_1_38793:@)
batch_normalization_1_38795:@)
batch_normalization_1_38797:@(
conv2d_2_38819:@@
conv2d_2_38821:@)
batch_normalization_2_38824:@)
batch_normalization_2_38826:@)
batch_normalization_2_38828:@)
batch_normalization_2_38830:@

pred_38866:@

pred_38868:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢pred/StatefulPartitionedCall¢-pred/kernel/Regularizer/Square/ReadVariableOpÔ
 img_augmentation/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38046
conv2d/StatefulPartitionedCallStatefulPartitionedCall)img_augmentation/PartitionedCall:output:0conv2d_38753conv2d_38755*
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
GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_38752ÿ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_38758batch_normalization_38760batch_normalization_38762batch_normalization_38764*
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
GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_38527ê
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
@__inference_re_lu_layer_call_and_return_conditional_losses_38772ä
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38578
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_38786conv2d_1_38788*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38785
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_38791batch_normalization_1_38793batch_normalization_1_38795batch_normalization_1_38797*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38603ð
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_38805ê
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38654
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_38819conv2d_2_38821*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_38818
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_38824batch_normalization_2_38826batch_normalization_2_38828batch_normalization_2_38830*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38679ð
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_38838Ô
avg_pool/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
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
GPU2*0J 8 *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_38731Û
top_dropout/PartitionedCallPartitionedCall!avg_pool/PartitionedCall:output:0*
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
F__inference_top_dropout_layer_call_and_return_conditional_losses_38846þ
pred/StatefulPartitionedCallStatefulPartitionedCall$top_dropout/PartitionedCall:output:0
pred_38866
pred_38868*
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
GPU2*0J 8 *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_38865x
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
pred_38866*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^pred/StatefulPartitionedCall.^pred/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÖN

G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39364
input_1$
img_augmentation_39294:	$
img_augmentation_39296:	$
img_augmentation_39298:	$
img_augmentation_39300:	&
conv2d_39303:		 
conv2d_39305: '
batch_normalization_39308: '
batch_normalization_39310: '
batch_normalization_39312: '
batch_normalization_39314: (
conv2d_1_39319: @
conv2d_1_39321:@)
batch_normalization_1_39324:@)
batch_normalization_1_39326:@)
batch_normalization_1_39328:@)
batch_normalization_1_39330:@(
conv2d_2_39335:@@
conv2d_2_39337:@)
batch_normalization_2_39340:@)
batch_normalization_2_39342:@)
batch_normalization_2_39344:@)
batch_normalization_2_39346:@

pred_39352:@

pred_39354:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢(img_augmentation/StatefulPartitionedCall¢pred/StatefulPartitionedCall¢-pred/kernel/Regularizer/Square/ReadVariableOp¢#top_dropout/StatefulPartitionedCallÉ
(img_augmentation/StatefulPartitionedCallStatefulPartitionedCallinput_1img_augmentation_39294img_augmentation_39296img_augmentation_39298img_augmentation_39300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38457
conv2d/StatefulPartitionedCallStatefulPartitionedCall1img_augmentation/StatefulPartitionedCall:output:0conv2d_39303conv2d_39305*
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
GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_38752ý
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_39308batch_normalization_39310batch_normalization_39312batch_normalization_39314*
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
GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_38558ê
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
@__inference_re_lu_layer_call_and_return_conditional_losses_38772ä
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38578
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_39319conv2d_1_39321*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38785
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_39324batch_normalization_1_39326batch_normalization_1_39328batch_normalization_1_39330*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38634ð
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_38805ê
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38654
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_39335conv2d_2_39337*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_38818
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_39340batch_normalization_2_39342batch_normalization_2_39344batch_normalization_2_39346*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38710ð
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_38838Ô
avg_pool/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
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
GPU2*0J 8 *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_38731ë
#top_dropout/StatefulPartitionedCallStatefulPartitionedCall!avg_pool/PartitionedCall:output:0*
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
F__inference_top_dropout_layer_call_and_return_conditional_losses_38951
pred/StatefulPartitionedCallStatefulPartitionedCall,top_dropout/StatefulPartitionedCall:output:0
pred_39352
pred_39354*
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
GPU2*0J 8 *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_38865x
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
pred_39352*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^img_augmentation/StatefulPartitionedCall^pred/StatefulPartitionedCall.^pred/kernel/Regularizer/Square/ReadVariableOp$^top_dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(img_augmentation/StatefulPartitionedCall(img_augmentation/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp2J
#top_dropout/StatefulPartitionedCall#top_dropout/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ô

4__inference_random_translation_1_layer_call_fn_40878

inputs
unknown:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38298y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»

$__inference_pred_layer_call_fn_40704

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCall×
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
GPU2*0J 8 *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_38865o
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
©

ú
A__inference_conv2d_layer_call_and_return_conditional_losses_40377

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

h
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38043

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
d
F__inference_top_dropout_layer_call_and_return_conditional_losses_38846

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
Ë

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_40522

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
§

ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40478

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
¤
G
+__inference_top_dropout_layer_call_fn_40667

inputs
identity´
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
GPU2*0J 8 *O
fJRH
F__inference_top_dropout_layer_call_and_return_conditional_losses_38846`
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
ëq
¤
 __inference__wrapped_model_38014
input_1L
2try_drop_fur_conv2d_conv2d_readvariableop_resource:		 A
3try_drop_fur_conv2d_biasadd_readvariableop_resource: F
8try_drop_fur_batch_normalization_readvariableop_resource: H
:try_drop_fur_batch_normalization_readvariableop_1_resource: W
Itry_drop_fur_batch_normalization_fusedbatchnormv3_readvariableop_resource: Y
Ktry_drop_fur_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: N
4try_drop_fur_conv2d_1_conv2d_readvariableop_resource: @C
5try_drop_fur_conv2d_1_biasadd_readvariableop_resource:@H
:try_drop_fur_batch_normalization_1_readvariableop_resource:@J
<try_drop_fur_batch_normalization_1_readvariableop_1_resource:@Y
Ktry_drop_fur_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@[
Mtry_drop_fur_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@N
4try_drop_fur_conv2d_2_conv2d_readvariableop_resource:@@C
5try_drop_fur_conv2d_2_biasadd_readvariableop_resource:@H
:try_drop_fur_batch_normalization_2_readvariableop_resource:@J
<try_drop_fur_batch_normalization_2_readvariableop_1_resource:@Y
Ktry_drop_fur_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@[
Mtry_drop_fur_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@B
0try_drop_fur_pred_matmul_readvariableop_resource:@?
1try_drop_fur_pred_biasadd_readvariableop_resource:
identity¢@try_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp¢Btry_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢/try_drop_fur/batch_normalization/ReadVariableOp¢1try_drop_fur/batch_normalization/ReadVariableOp_1¢Btry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢Dtry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢1try_drop_fur/batch_normalization_1/ReadVariableOp¢3try_drop_fur/batch_normalization_1/ReadVariableOp_1¢Btry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢Dtry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢1try_drop_fur/batch_normalization_2/ReadVariableOp¢3try_drop_fur/batch_normalization_2/ReadVariableOp_1¢*try_drop_fur/conv2d/BiasAdd/ReadVariableOp¢)try_drop_fur/conv2d/Conv2D/ReadVariableOp¢,try_drop_fur/conv2d_1/BiasAdd/ReadVariableOp¢+try_drop_fur/conv2d_1/Conv2D/ReadVariableOp¢,try_drop_fur/conv2d_2/BiasAdd/ReadVariableOp¢+try_drop_fur/conv2d_2/Conv2D/ReadVariableOp¢(try_drop_fur/pred/BiasAdd/ReadVariableOp¢'try_drop_fur/pred/MatMul/ReadVariableOp¤
)try_drop_fur/conv2d/Conv2D/ReadVariableOpReadVariableOp2try_drop_fur_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0Ã
try_drop_fur/conv2d/Conv2DConv2Dinput_11try_drop_fur/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides

*try_drop_fur/conv2d/BiasAdd/ReadVariableOpReadVariableOp3try_drop_fur_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¹
try_drop_fur/conv2d/BiasAddBiasAdd#try_drop_fur/conv2d/Conv2D:output:02try_drop_fur/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? ¤
/try_drop_fur/batch_normalization/ReadVariableOpReadVariableOp8try_drop_fur_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0¨
1try_drop_fur/batch_normalization/ReadVariableOp_1ReadVariableOp:try_drop_fur_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0Æ
@try_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpItry_drop_fur_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ê
Btry_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKtry_drop_fur_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ù
1try_drop_fur/batch_normalization/FusedBatchNormV3FusedBatchNormV3$try_drop_fur/conv2d/BiasAdd:output:07try_drop_fur/batch_normalization/ReadVariableOp:value:09try_drop_fur/batch_normalization/ReadVariableOp_1:value:0Htry_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jtry_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ?? : : : : :*
epsilon%o:*
is_training( 
try_drop_fur/re_lu/ReluRelu5try_drop_fur/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? Á
"try_drop_fur/max_pooling2d/MaxPoolMaxPool%try_drop_fur/re_lu/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
¨
+try_drop_fur/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4try_drop_fur_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ë
try_drop_fur/conv2d_1/Conv2DConv2D+try_drop_fur/max_pooling2d/MaxPool:output:03try_drop_fur/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides

,try_drop_fur/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5try_drop_fur_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¿
try_drop_fur/conv2d_1/BiasAddBiasAdd%try_drop_fur/conv2d_1/Conv2D:output:04try_drop_fur/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@¨
1try_drop_fur/batch_normalization_1/ReadVariableOpReadVariableOp:try_drop_fur_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3try_drop_fur/batch_normalization_1/ReadVariableOp_1ReadVariableOp<try_drop_fur_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ê
Btry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKtry_drop_fur_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Î
Dtry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMtry_drop_fur_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
3try_drop_fur/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&try_drop_fur/conv2d_1/BiasAdd:output:09try_drop_fur/batch_normalization_1/ReadVariableOp:value:0;try_drop_fur/batch_normalization_1/ReadVariableOp_1:value:0Jtry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Ltry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ		@:@:@:@:@:*
epsilon%o:*
is_training( 
try_drop_fur/re_lu_1/ReluRelu7try_drop_fur/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@Å
$try_drop_fur/max_pooling2d_1/MaxPoolMaxPool'try_drop_fur/re_lu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
¨
+try_drop_fur/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4try_drop_fur_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0í
try_drop_fur/conv2d_2/Conv2DConv2D-try_drop_fur/max_pooling2d_1/MaxPool:output:03try_drop_fur/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

,try_drop_fur/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5try_drop_fur_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¿
try_drop_fur/conv2d_2/BiasAddBiasAdd%try_drop_fur/conv2d_2/Conv2D:output:04try_drop_fur/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
1try_drop_fur/batch_normalization_2/ReadVariableOpReadVariableOp:try_drop_fur_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0¬
3try_drop_fur/batch_normalization_2/ReadVariableOp_1ReadVariableOp<try_drop_fur_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ê
Btry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKtry_drop_fur_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Î
Dtry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMtry_drop_fur_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
3try_drop_fur/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&try_drop_fur/conv2d_2/BiasAdd:output:09try_drop_fur/batch_normalization_2/ReadVariableOp:value:0;try_drop_fur/batch_normalization_2/ReadVariableOp_1:value:0Jtry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Ltry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
try_drop_fur/re_lu_2/ReluRelu7try_drop_fur/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
+try_drop_fur/avg_pool/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ±
try_drop_fur/avg_pool/MaxMax'try_drop_fur/re_lu_2/Relu:activations:04try_drop_fur/avg_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!try_drop_fur/top_dropout/IdentityIdentity"try_drop_fur/avg_pool/Max:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'try_drop_fur/pred/MatMul/ReadVariableOpReadVariableOp0try_drop_fur_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0±
try_drop_fur/pred/MatMulMatMul*try_drop_fur/top_dropout/Identity:output:0/try_drop_fur/pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(try_drop_fur/pred/BiasAdd/ReadVariableOpReadVariableOp1try_drop_fur_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
try_drop_fur/pred/BiasAddBiasAdd"try_drop_fur/pred/MatMul:product:00try_drop_fur/pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
try_drop_fur/pred/SoftmaxSoftmax"try_drop_fur/pred/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
IdentityIdentity#try_drop_fur/pred/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
NoOpNoOpA^try_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOpC^try_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp_10^try_drop_fur/batch_normalization/ReadVariableOp2^try_drop_fur/batch_normalization/ReadVariableOp_1C^try_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^try_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12^try_drop_fur/batch_normalization_1/ReadVariableOp4^try_drop_fur/batch_normalization_1/ReadVariableOp_1C^try_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^try_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^try_drop_fur/batch_normalization_2/ReadVariableOp4^try_drop_fur/batch_normalization_2/ReadVariableOp_1+^try_drop_fur/conv2d/BiasAdd/ReadVariableOp*^try_drop_fur/conv2d/Conv2D/ReadVariableOp-^try_drop_fur/conv2d_1/BiasAdd/ReadVariableOp,^try_drop_fur/conv2d_1/Conv2D/ReadVariableOp-^try_drop_fur/conv2d_2/BiasAdd/ReadVariableOp,^try_drop_fur/conv2d_2/Conv2D/ReadVariableOp)^try_drop_fur/pred/BiasAdd/ReadVariableOp(^try_drop_fur/pred/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2
@try_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp@try_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp2
Btry_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Btry_drop_fur/batch_normalization/FusedBatchNormV3/ReadVariableOp_12b
/try_drop_fur/batch_normalization/ReadVariableOp/try_drop_fur/batch_normalization/ReadVariableOp2f
1try_drop_fur/batch_normalization/ReadVariableOp_11try_drop_fur/batch_normalization/ReadVariableOp_12
Btry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOpBtry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
Dtry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Dtry_drop_fur/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12f
1try_drop_fur/batch_normalization_1/ReadVariableOp1try_drop_fur/batch_normalization_1/ReadVariableOp2j
3try_drop_fur/batch_normalization_1/ReadVariableOp_13try_drop_fur/batch_normalization_1/ReadVariableOp_12
Btry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBtry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2
Dtry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dtry_drop_fur/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1try_drop_fur/batch_normalization_2/ReadVariableOp1try_drop_fur/batch_normalization_2/ReadVariableOp2j
3try_drop_fur/batch_normalization_2/ReadVariableOp_13try_drop_fur/batch_normalization_2/ReadVariableOp_12X
*try_drop_fur/conv2d/BiasAdd/ReadVariableOp*try_drop_fur/conv2d/BiasAdd/ReadVariableOp2V
)try_drop_fur/conv2d/Conv2D/ReadVariableOp)try_drop_fur/conv2d/Conv2D/ReadVariableOp2\
,try_drop_fur/conv2d_1/BiasAdd/ReadVariableOp,try_drop_fur/conv2d_1/BiasAdd/ReadVariableOp2Z
+try_drop_fur/conv2d_1/Conv2D/ReadVariableOp+try_drop_fur/conv2d_1/Conv2D/ReadVariableOp2\
,try_drop_fur/conv2d_2/BiasAdd/ReadVariableOp,try_drop_fur/conv2d_2/BiasAdd/ReadVariableOp2Z
+try_drop_fur/conv2d_2/Conv2D/ReadVariableOp+try_drop_fur/conv2d_2/Conv2D/ReadVariableOp2T
(try_drop_fur/pred/BiasAdd/ReadVariableOp(try_drop_fur/pred/BiasAdd/ReadVariableOp2R
'try_drop_fur/pred/MatMul/ReadVariableOp'try_drop_fur/pred/MatMul/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

g
K__inference_img_augmentation_layer_call_and_return_conditional_losses_40037

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

1__inference_random_contrast_1_layer_call_fn_41084

inputs
unknown:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38098y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40623

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
ÐÚ
½
K__inference_img_augmentation_layer_call_and_return_conditional_losses_40358

inputsH
:random_rotation_1_stateful_uniform_rngreadandskip_resource:	K
=random_translation_1_stateful_uniform_rngreadandskip_resource:	M
?random_flip_1_stateful_uniform_full_int_rngreadandskip_resource:	Q
Crandom_contrast_1_stateful_uniform_full_int_rngreadandskip_resource:	
identity¢:random_contrast_1/stateful_uniform_full_int/RngReadAndSkip¢6random_flip_1/stateful_uniform_full_int/RngReadAndSkip¢1random_rotation_1/stateful_uniform/RngReadAndSkip¢4random_translation_1/stateful_uniform/RngReadAndSkip¢6random_translation_1/stateful_uniform_1/RngReadAndSkipM
random_rotation_1/ShapeShapeinputs*
T0*
_output_shapes
:o
%random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
random_rotation_1/strided_sliceStridedSlice random_rotation_1/Shape:output:0.random_rotation_1/strided_slice/stack:output:00random_rotation_1/strided_slice/stack_1:output:00random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
'random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ|
)random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿs
)random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
!random_rotation_1/strided_slice_1StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_1/stack:output:02random_rotation_1/strided_slice_1/stack_1:output:02random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
random_rotation_1/CastCast*random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: z
'random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ|
)random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿs
)random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
!random_rotation_1/strided_slice_2StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_2/stack:output:02random_rotation_1/strided_slice_2/stack_1:output:02random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
random_rotation_1/Cast_1Cast*random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 
(random_rotation_1/stateful_uniform/shapePack(random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:k
&random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq¿k
&random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?r
(random_rotation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¶
'random_rotation_1/stateful_uniform/ProdProd1random_rotation_1/stateful_uniform/shape:output:01random_rotation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: k
)random_rotation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
)random_rotation_1/stateful_uniform/Cast_1Cast0random_rotation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: þ
1random_rotation_1/stateful_uniform/RngReadAndSkipRngReadAndSkip:random_rotation_1_stateful_uniform_rngreadandskip_resource2random_rotation_1/stateful_uniform/Cast/x:output:0-random_rotation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:
6random_rotation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8random_rotation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8random_rotation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0random_rotation_1/stateful_uniform/strided_sliceStridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0?random_rotation_1/stateful_uniform/strided_slice/stack:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_1:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask¡
*random_rotation_1/stateful_uniform/BitcastBitcast9random_rotation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
8random_rotation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
:random_rotation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:random_rotation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ü
2random_rotation_1/stateful_uniform/strided_slice_1StridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Arandom_rotation_1/stateful_uniform/strided_slice_1/stack:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_1:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¥
,random_rotation_1/stateful_uniform/Bitcast_1Bitcast;random_rotation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
?random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :å
;random_rotation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV21random_rotation_1/stateful_uniform/shape:output:05random_rotation_1/stateful_uniform/Bitcast_1:output:03random_rotation_1/stateful_uniform/Bitcast:output:0Hrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
&random_rotation_1/stateful_uniform/subSub/random_rotation_1/stateful_uniform/max:output:0/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: Í
&random_rotation_1/stateful_uniform/mulMulDrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2:output:0*random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
"random_rotation_1/stateful_uniformAddV2*random_rotation_1/stateful_uniform/mul:z:0/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
'random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
%random_rotation_1/rotation_matrix/subSubrandom_rotation_1/Cast_1:y:00random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 
%random_rotation_1/rotation_matrix/CosCos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
'random_rotation_1/rotation_matrix/sub_1Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: ²
%random_rotation_1/rotation_matrix/mulMul)random_rotation_1/rotation_matrix/Cos:y:0+random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%random_rotation_1/rotation_matrix/SinSin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'random_rotation_1/rotation_matrix/sub_2Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ´
'random_rotation_1/rotation_matrix/mul_1Mul)random_rotation_1/rotation_matrix/Sin:y:0+random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
'random_rotation_1/rotation_matrix/sub_3Sub)random_rotation_1/rotation_matrix/mul:z:0+random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
'random_rotation_1/rotation_matrix/sub_4Sub)random_rotation_1/rotation_matrix/sub:z:0+random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Å
)random_rotation_1/rotation_matrix/truedivRealDiv+random_rotation_1/rotation_matrix/sub_4:z:04random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'random_rotation_1/rotation_matrix/sub_5Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 
'random_rotation_1/rotation_matrix/Sin_1Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
'random_rotation_1/rotation_matrix/sub_6Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ¶
'random_rotation_1/rotation_matrix/mul_2Mul+random_rotation_1/rotation_matrix/Sin_1:y:0+random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'random_rotation_1/rotation_matrix/Cos_1Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'random_rotation_1/rotation_matrix/sub_7Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ¶
'random_rotation_1/rotation_matrix/mul_3Mul+random_rotation_1/rotation_matrix/Cos_1:y:0+random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
%random_rotation_1/rotation_matrix/addAddV2+random_rotation_1/rotation_matrix/mul_2:z:0+random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
'random_rotation_1/rotation_matrix/sub_8Sub+random_rotation_1/rotation_matrix/sub_5:z:0)random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
-random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @É
+random_rotation_1/rotation_matrix/truediv_1RealDiv+random_rotation_1/rotation_matrix/sub_8:z:06random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
'random_rotation_1/rotation_matrix/ShapeShape&random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:
5random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
/random_rotation_1/rotation_matrix/strided_sliceStridedSlice0random_rotation_1/rotation_matrix/Shape:output:0>random_rotation_1/rotation_matrix/strided_slice/stack:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'random_rotation_1/rotation_matrix/Cos_2Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_1/rotation_matrix/strided_slice_1StridedSlice+random_rotation_1/rotation_matrix/Cos_2:y:0@random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
'random_rotation_1/rotation_matrix/Sin_2Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_1/rotation_matrix/strided_slice_2StridedSlice+random_rotation_1/rotation_matrix/Sin_2:y:0@random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
%random_rotation_1/rotation_matrix/NegNeg:random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      °
1random_rotation_1/rotation_matrix/strided_slice_3StridedSlice-random_rotation_1/rotation_matrix/truediv:z:0@random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
'random_rotation_1/rotation_matrix/Sin_3Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_1/rotation_matrix/strided_slice_4StridedSlice+random_rotation_1/rotation_matrix/Sin_3:y:0@random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
'random_rotation_1/rotation_matrix/Cos_3Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_1/rotation_matrix/strided_slice_5StridedSlice+random_rotation_1/rotation_matrix/Cos_3:y:0@random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
7random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ²
1random_rotation_1/rotation_matrix/strided_slice_6StridedSlice/random_rotation_1/rotation_matrix/truediv_1:z:0@random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskr
0random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ù
.random_rotation_1/rotation_matrix/zeros/packedPack8random_rotation_1/rotation_matrix/strided_slice:output:09random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:r
-random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ò
'random_rotation_1/rotation_matrix/zerosFill7random_rotation_1/rotation_matrix/zeros/packed:output:06random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
(random_rotation_1/rotation_matrix/concatConcatV2:random_rotation_1/rotation_matrix/strided_slice_1:output:0)random_rotation_1/rotation_matrix/Neg:y:0:random_rotation_1/rotation_matrix/strided_slice_3:output:0:random_rotation_1/rotation_matrix/strided_slice_4:output:0:random_rotation_1/rotation_matrix/strided_slice_5:output:0:random_rotation_1/rotation_matrix/strided_slice_6:output:00random_rotation_1/rotation_matrix/zeros:output:06random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
!random_rotation_1/transform/ShapeShapeinputs*
T0*
_output_shapes
:y
/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
)random_rotation_1/transform/strided_sliceStridedSlice*random_rotation_1/transform/Shape:output:08random_rotation_1/transform/strided_slice/stack:output:0:random_rotation_1/transform/strided_slice/stack_1:output:0:random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:k
&random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    é
6random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputs1random_rotation_1/rotation_matrix/concat:output:02random_rotation_1/transform/strided_slice:output:0/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
random_translation_1/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:r
(random_translation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*random_translation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*random_translation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"random_translation_1/strided_sliceStridedSlice#random_translation_1/Shape:output:01random_translation_1/strided_slice/stack:output:03random_translation_1/strided_slice/stack_1:output:03random_translation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
*random_translation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
,random_translation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿv
,random_translation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Â
$random_translation_1/strided_slice_1StridedSlice#random_translation_1/Shape:output:03random_translation_1/strided_slice_1/stack:output:05random_translation_1/strided_slice_1/stack_1:output:05random_translation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
random_translation_1/CastCast-random_translation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: }
*random_translation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
,random_translation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿv
,random_translation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Â
$random_translation_1/strided_slice_2StridedSlice#random_translation_1/Shape:output:03random_translation_1/strided_slice_2/stack:output:05random_translation_1/strided_slice_2/stack_1:output:05random_translation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
random_translation_1/Cast_1Cast-random_translation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: o
-random_translation_1/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Æ
+random_translation_1/stateful_uniform/shapePack+random_translation_1/strided_slice:output:06random_translation_1/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:n
)random_translation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½n
)random_translation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=u
+random_translation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¿
*random_translation_1/stateful_uniform/ProdProd4random_translation_1/stateful_uniform/shape:output:04random_translation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: n
,random_translation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
,random_translation_1/stateful_uniform/Cast_1Cast3random_translation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
4random_translation_1/stateful_uniform/RngReadAndSkipRngReadAndSkip=random_translation_1_stateful_uniform_rngreadandskip_resource5random_translation_1/stateful_uniform/Cast/x:output:00random_translation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:
9random_translation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;random_translation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;random_translation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3random_translation_1/stateful_uniform/strided_sliceStridedSlice<random_translation_1/stateful_uniform/RngReadAndSkip:value:0Brandom_translation_1/stateful_uniform/strided_slice/stack:output:0Drandom_translation_1/stateful_uniform/strided_slice/stack_1:output:0Drandom_translation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask§
-random_translation_1/stateful_uniform/BitcastBitcast<random_translation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
;random_translation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
=random_translation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_translation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_translation_1/stateful_uniform/strided_slice_1StridedSlice<random_translation_1/stateful_uniform/RngReadAndSkip:value:0Drandom_translation_1/stateful_uniform/strided_slice_1/stack:output:0Frandom_translation_1/stateful_uniform/strided_slice_1/stack_1:output:0Frandom_translation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:«
/random_translation_1/stateful_uniform/Bitcast_1Bitcast>random_translation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Brandom_translation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ø
>random_translation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV24random_translation_1/stateful_uniform/shape:output:08random_translation_1/stateful_uniform/Bitcast_1:output:06random_translation_1/stateful_uniform/Bitcast:output:0Krandom_translation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
)random_translation_1/stateful_uniform/subSub2random_translation_1/stateful_uniform/max:output:02random_translation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: Ú
)random_translation_1/stateful_uniform/mulMulGrandom_translation_1/stateful_uniform/StatelessRandomUniformV2:output:0-random_translation_1/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
%random_translation_1/stateful_uniformAddV2-random_translation_1/stateful_uniform/mul:z:02random_translation_1/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
random_translation_1/mulMul)random_translation_1/stateful_uniform:z:0random_translation_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/random_translation_1/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ê
-random_translation_1/stateful_uniform_1/shapePack+random_translation_1/strided_slice:output:08random_translation_1/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:p
+random_translation_1/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½p
+random_translation_1/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=w
-random_translation_1/stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
,random_translation_1/stateful_uniform_1/ProdProd6random_translation_1/stateful_uniform_1/shape:output:06random_translation_1/stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: p
.random_translation_1/stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
.random_translation_1/stateful_uniform_1/Cast_1Cast5random_translation_1/stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ç
6random_translation_1/stateful_uniform_1/RngReadAndSkipRngReadAndSkip=random_translation_1_stateful_uniform_rngreadandskip_resource7random_translation_1/stateful_uniform_1/Cast/x:output:02random_translation_1/stateful_uniform_1/Cast_1:y:05^random_translation_1/stateful_uniform/RngReadAndSkip*
_output_shapes
:
;random_translation_1/stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_translation_1/stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_translation_1/stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_translation_1/stateful_uniform_1/strided_sliceStridedSlice>random_translation_1/stateful_uniform_1/RngReadAndSkip:value:0Drandom_translation_1/stateful_uniform_1/strided_slice/stack:output:0Frandom_translation_1/stateful_uniform_1/strided_slice/stack_1:output:0Frandom_translation_1/stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask«
/random_translation_1/stateful_uniform_1/BitcastBitcast>random_translation_1/stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
=random_translation_1/stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_translation_1/stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_translation_1/stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7random_translation_1/stateful_uniform_1/strided_slice_1StridedSlice>random_translation_1/stateful_uniform_1/RngReadAndSkip:value:0Frandom_translation_1/stateful_uniform_1/strided_slice_1/stack:output:0Hrandom_translation_1/stateful_uniform_1/strided_slice_1/stack_1:output:0Hrandom_translation_1/stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¯
1random_translation_1/stateful_uniform_1/Bitcast_1Bitcast@random_translation_1/stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Drandom_translation_1/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
@random_translation_1/stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV26random_translation_1/stateful_uniform_1/shape:output:0:random_translation_1/stateful_uniform_1/Bitcast_1:output:08random_translation_1/stateful_uniform_1/Bitcast:output:0Mrandom_translation_1/stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
+random_translation_1/stateful_uniform_1/subSub4random_translation_1/stateful_uniform_1/max:output:04random_translation_1/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: à
+random_translation_1/stateful_uniform_1/mulMulIrandom_translation_1/stateful_uniform_1/StatelessRandomUniformV2:output:0/random_translation_1/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
'random_translation_1/stateful_uniform_1AddV2/random_translation_1/stateful_uniform_1/mul:z:04random_translation_1/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
random_translation_1/mul_1Mul+random_translation_1/stateful_uniform_1:z:0random_translation_1/Cast_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 random_translation_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
random_translation_1/concatConcatV2random_translation_1/mul_1:z:0random_translation_1/mul:z:0)random_translation_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-random_translation_1/translation_matrix/ShapeShape$random_translation_1/concat:output:0*
T0*
_output_shapes
:
;random_translation_1/translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_translation_1/translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_translation_1/translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_translation_1/translation_matrix/strided_sliceStridedSlice6random_translation_1/translation_matrix/Shape:output:0Drandom_translation_1/translation_matrix/strided_slice/stack:output:0Frandom_translation_1/translation_matrix/strided_slice/stack_1:output:0Frandom_translation_1/translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5random_translation_1/translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :é
3random_translation_1/translation_matrix/ones/packedPack>random_translation_1/translation_matrix/strided_slice:output:0>random_translation_1/translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:w
2random_translation_1/translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?á
,random_translation_1/translation_matrix/onesFill<random_translation_1/translation_matrix/ones/packed:output:0;random_translation_1/translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
6random_translation_1/translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ë
4random_translation_1/translation_matrix/zeros/packedPack>random_translation_1/translation_matrix/strided_slice:output:0?random_translation_1/translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:x
3random_translation_1/translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ä
-random_translation_1/translation_matrix/zerosFill=random_translation_1/translation_matrix/zeros/packed:output:0<random_translation_1/translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_translation_1/translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?random_translation_1/translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?random_translation_1/translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ×
7random_translation_1/translation_matrix/strided_slice_1StridedSlice$random_translation_1/concat:output:0Frandom_translation_1/translation_matrix/strided_slice_1/stack:output:0Hrandom_translation_1/translation_matrix/strided_slice_1/stack_1:output:0Hrandom_translation_1/translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask¦
+random_translation_1/translation_matrix/NegNeg@random_translation_1/translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8random_translation_1/translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ï
6random_translation_1/translation_matrix/zeros_1/packedPack>random_translation_1/translation_matrix/strided_slice:output:0Arandom_translation_1/translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:z
5random_translation_1/translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ê
/random_translation_1/translation_matrix/zeros_1Fill?random_translation_1/translation_matrix/zeros_1/packed:output:0>random_translation_1/translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7random_translation_1/translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :í
5random_translation_1/translation_matrix/ones_1/packedPack>random_translation_1/translation_matrix/strided_slice:output:0@random_translation_1/translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:y
4random_translation_1/translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ç
.random_translation_1/translation_matrix/ones_1Fill>random_translation_1/translation_matrix/ones_1/packed:output:0=random_translation_1/translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_translation_1/translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?random_translation_1/translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?random_translation_1/translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ×
7random_translation_1/translation_matrix/strided_slice_2StridedSlice$random_translation_1/concat:output:0Frandom_translation_1/translation_matrix/strided_slice_2/stack:output:0Hrandom_translation_1/translation_matrix/strided_slice_2/stack_1:output:0Hrandom_translation_1/translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask¨
-random_translation_1/translation_matrix/Neg_1Neg@random_translation_1/translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8random_translation_1/translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ï
6random_translation_1/translation_matrix/zeros_2/packedPack>random_translation_1/translation_matrix/strided_slice:output:0Arandom_translation_1/translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:z
5random_translation_1/translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ê
/random_translation_1/translation_matrix/zeros_2Fill?random_translation_1/translation_matrix/zeros_2/packed:output:0>random_translation_1/translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
3random_translation_1/translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :³
.random_translation_1/translation_matrix/concatConcatV25random_translation_1/translation_matrix/ones:output:06random_translation_1/translation_matrix/zeros:output:0/random_translation_1/translation_matrix/Neg:y:08random_translation_1/translation_matrix/zeros_1:output:07random_translation_1/translation_matrix/ones_1:output:01random_translation_1/translation_matrix/Neg_1:y:08random_translation_1/translation_matrix/zeros_2:output:0<random_translation_1/translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$random_translation_1/transform/ShapeShapeKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:|
2random_translation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4random_translation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4random_translation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
,random_translation_1/transform/strided_sliceStridedSlice-random_translation_1/transform/Shape:output:0;random_translation_1/transform/strided_slice/stack:output:0=random_translation_1/transform/strided_slice/stack_1:output:0=random_translation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:n
)random_translation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
9random_translation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Krandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:07random_translation_1/translation_matrix/concat:output:05random_translation_1/transform/strided_slice:output:02random_translation_1/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARw
-random_flip_1/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:w
-random_flip_1/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
,random_flip_1/stateful_uniform_full_int/ProdProd6random_flip_1/stateful_uniform_full_int/shape:output:06random_flip_1/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: p
.random_flip_1/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
.random_flip_1/stateful_uniform_full_int/Cast_1Cast5random_flip_1/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
6random_flip_1/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip?random_flip_1_stateful_uniform_full_int_rngreadandskip_resource7random_flip_1/stateful_uniform_full_int/Cast/x:output:02random_flip_1/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
;random_flip_1/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_flip_1/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_flip_1/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_flip_1/stateful_uniform_full_int/strided_sliceStridedSlice>random_flip_1/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip_1/stateful_uniform_full_int/strided_slice/stack:output:0Frandom_flip_1/stateful_uniform_full_int/strided_slice/stack_1:output:0Frandom_flip_1/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask«
/random_flip_1/stateful_uniform_full_int/BitcastBitcast>random_flip_1/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
=random_flip_1/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_flip_1/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_flip_1/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7random_flip_1/stateful_uniform_full_int/strided_slice_1StridedSlice>random_flip_1/stateful_uniform_full_int/RngReadAndSkip:value:0Frandom_flip_1/stateful_uniform_full_int/strided_slice_1/stack:output:0Hrandom_flip_1/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Hrandom_flip_1/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¯
1random_flip_1/stateful_uniform_full_int/Bitcast_1Bitcast@random_flip_1/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0m
+random_flip_1/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :×
'random_flip_1/stateful_uniform_full_intStatelessRandomUniformFullIntV26random_flip_1/stateful_uniform_full_int/shape:output:0:random_flip_1/stateful_uniform_full_int/Bitcast_1:output:08random_flip_1/stateful_uniform_full_int/Bitcast:output:04random_flip_1/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	b
random_flip_1/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ¢
random_flip_1/stackPack0random_flip_1/stateful_uniform_full_int:output:0!random_flip_1/zeros_like:output:0*
N*
T0	*
_output_shapes

:r
!random_flip_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#random_flip_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#random_flip_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
random_flip_1/strided_sliceStridedSlicerandom_flip_1/stack:output:0*random_flip_1/strided_slice/stack:output:0,random_flip_1/strided_slice/stack_1:output:0,random_flip_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask§
Arandom_flip_1/stateless_random_flip_left_right/control_dependencyIdentityNrandom_translation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*L
_classB
@>loc:@random_translation_1/transform/ImageProjectiveTransformV3*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4random_flip_1/stateless_random_flip_left_right/ShapeShapeJrandom_flip_1/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Brandom_flip_1/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Drandom_flip_1/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Drandom_flip_1/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
<random_flip_1/stateless_random_flip_left_right/strided_sliceStridedSlice=random_flip_1/stateless_random_flip_left_right/Shape:output:0Krandom_flip_1/stateless_random_flip_left_right/strided_slice/stack:output:0Mrandom_flip_1/stateless_random_flip_left_right/strided_slice/stack_1:output:0Mrandom_flip_1/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÊ
Mrandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/shapePackErandom_flip_1/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
Krandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Krandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ì
drandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip_1/strided_slice:output:0* 
_output_shapes
::¦
drandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ä
`random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Vrandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0jrandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0nrandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0mrandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Krandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/subSubTrandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Trandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¼
Krandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/mulMulirandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Orandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Grandom_flip_1/stateless_random_flip_left_right/stateless_random_uniformAddV2Orandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Trandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>random_flip_1/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>random_flip_1/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
>random_flip_1/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
<random_flip_1/stateless_random_flip_left_right/Reshape/shapePackErandom_flip_1/stateless_random_flip_left_right/strided_slice:output:0Grandom_flip_1/stateless_random_flip_left_right/Reshape/shape/1:output:0Grandom_flip_1/stateless_random_flip_left_right/Reshape/shape/2:output:0Grandom_flip_1/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
6random_flip_1/stateless_random_flip_left_right/ReshapeReshapeKrandom_flip_1/stateless_random_flip_left_right/stateless_random_uniform:z:0Erandom_flip_1/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
4random_flip_1/stateless_random_flip_left_right/RoundRound?random_flip_1/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_flip_1/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
8random_flip_1/stateless_random_flip_left_right/ReverseV2	ReverseV2Jrandom_flip_1/stateless_random_flip_left_right/control_dependency:output:0Frandom_flip_1/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
2random_flip_1/stateless_random_flip_left_right/mulMul8random_flip_1/stateless_random_flip_left_right/Round:y:0Arandom_flip_1/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
4random_flip_1/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ì
2random_flip_1/stateless_random_flip_left_right/subSub=random_flip_1/stateless_random_flip_left_right/sub/x:output:08random_flip_1/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
4random_flip_1/stateless_random_flip_left_right/mul_1Mul6random_flip_1/stateless_random_flip_left_right/sub:z:0Jrandom_flip_1/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
2random_flip_1/stateless_random_flip_left_right/addAddV26random_flip_1/stateless_random_flip_left_right/mul:z:08random_flip_1/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
1random_contrast_1/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:{
1random_contrast_1/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ñ
0random_contrast_1/stateful_uniform_full_int/ProdProd:random_contrast_1/stateful_uniform_full_int/shape:output:0:random_contrast_1/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: t
2random_contrast_1/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¥
2random_contrast_1/stateful_uniform_full_int/Cast_1Cast9random_contrast_1/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¢
:random_contrast_1/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipCrandom_contrast_1_stateful_uniform_full_int_rngreadandskip_resource;random_contrast_1/stateful_uniform_full_int/Cast/x:output:06random_contrast_1/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
?random_contrast_1/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Arandom_contrast_1/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Arandom_contrast_1/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
9random_contrast_1/stateful_uniform_full_int/strided_sliceStridedSliceBrandom_contrast_1/stateful_uniform_full_int/RngReadAndSkip:value:0Hrandom_contrast_1/stateful_uniform_full_int/strided_slice/stack:output:0Jrandom_contrast_1/stateful_uniform_full_int/strided_slice/stack_1:output:0Jrandom_contrast_1/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask³
3random_contrast_1/stateful_uniform_full_int/BitcastBitcastBrandom_contrast_1/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Arandom_contrast_1/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Crandom_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Crandom_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
;random_contrast_1/stateful_uniform_full_int/strided_slice_1StridedSliceBrandom_contrast_1/stateful_uniform_full_int/RngReadAndSkip:value:0Jrandom_contrast_1/stateful_uniform_full_int/strided_slice_1/stack:output:0Lrandom_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Lrandom_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:·
5random_contrast_1/stateful_uniform_full_int/Bitcast_1BitcastDrandom_contrast_1/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0q
/random_contrast_1/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :ë
+random_contrast_1/stateful_uniform_full_intStatelessRandomUniformFullIntV2:random_contrast_1/stateful_uniform_full_int/shape:output:0>random_contrast_1/stateful_uniform_full_int/Bitcast_1:output:0<random_contrast_1/stateful_uniform_full_int/Bitcast:output:08random_contrast_1/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	f
random_contrast_1/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ®
random_contrast_1/stackPack4random_contrast_1/stateful_uniform_full_int:output:0%random_contrast_1/zeros_like:output:0*
N*
T0	*
_output_shapes

:v
%random_contrast_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'random_contrast_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'random_contrast_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
random_contrast_1/strided_sliceStridedSlice random_contrast_1/stack:output:0.random_contrast_1/strided_slice/stack:output:00random_contrast_1/strided_slice/stack_1:output:00random_contrast_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_masks
0random_contrast_1/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB s
.random_contrast_1/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?s
.random_contrast_1/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?³
Grandom_contrast_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter(random_contrast_1/strided_slice:output:0* 
_output_shapes
::
Grandom_contrast_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :¦
Crandom_contrast_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV29random_contrast_1/stateless_random_uniform/shape:output:0Mrandom_contrast_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Qrandom_contrast_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Prandom_contrast_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: È
.random_contrast_1/stateless_random_uniform/subSub7random_contrast_1/stateless_random_uniform/max:output:07random_contrast_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ø
.random_contrast_1/stateless_random_uniform/mulMulLrandom_contrast_1/stateless_random_uniform/StatelessRandomUniformV2:output:02random_contrast_1/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: Á
*random_contrast_1/stateless_random_uniformAddV22random_contrast_1/stateless_random_uniform/mul:z:07random_contrast_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ð
!random_contrast_1/adjust_contrastAdjustContrastv26random_flip_1/stateless_random_flip_left_right/add:z:0.random_contrast_1/stateless_random_uniform:z:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*random_contrast_1/adjust_contrast/IdentityIdentity*random_contrast_1/adjust_contrast:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity3random_contrast_1/adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp;^random_contrast_1/stateful_uniform_full_int/RngReadAndSkip7^random_flip_1/stateful_uniform_full_int/RngReadAndSkip2^random_rotation_1/stateful_uniform/RngReadAndSkip5^random_translation_1/stateful_uniform/RngReadAndSkip7^random_translation_1/stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : 2x
:random_contrast_1/stateful_uniform_full_int/RngReadAndSkip:random_contrast_1/stateful_uniform_full_int/RngReadAndSkip2p
6random_flip_1/stateful_uniform_full_int/RngReadAndSkip6random_flip_1/stateful_uniform_full_int/RngReadAndSkip2f
1random_rotation_1/stateful_uniform/RngReadAndSkip1random_rotation_1/stateful_uniform/RngReadAndSkip2l
4random_translation_1/stateful_uniform/RngReadAndSkip4random_translation_1/stateful_uniform/RngReadAndSkip2p
6random_translation_1/stateful_uniform_1/RngReadAndSkip6random_translation_1/stateful_uniform_1/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
\
@__inference_re_lu_layer_call_and_return_conditional_losses_38772

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
ô
Ü
0__inference_img_augmentation_layer_call_fn_38481
random_rotation_1_input
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_rotation_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38457y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1
_user_specified_namerandom_rotation_1_input
É

N__inference_batch_normalization_layer_call_and_return_conditional_losses_40421

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
¸
Å
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38505
random_rotation_1_input%
random_rotation_1_38492:	(
random_translation_1_38495:	!
random_flip_1_38498:	%
random_contrast_1_38501:	
identity¢)random_contrast_1/StatefulPartitionedCall¢%random_flip_1/StatefulPartitionedCall¢)random_rotation_1/StatefulPartitionedCall¢,random_translation_1/StatefulPartitionedCall
)random_rotation_1/StatefulPartitionedCallStatefulPartitionedCallrandom_rotation_1_inputrandom_rotation_1_38492*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38429µ
,random_translation_1/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_1/StatefulPartitionedCall:output:0random_translation_1_38495*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38298£
%random_flip_1/StatefulPartitionedCallStatefulPartitionedCall5random_translation_1/StatefulPartitionedCall:output:0random_flip_1_38498*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38169¨
)random_contrast_1/StatefulPartitionedCallStatefulPartitionedCall.random_flip_1/StatefulPartitionedCall:output:0random_contrast_1_38501*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38098
IdentityIdentity2random_contrast_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
NoOpNoOp*^random_contrast_1/StatefulPartitionedCall&^random_flip_1/StatefulPartitionedCall*^random_rotation_1/StatefulPartitionedCall-^random_translation_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : 2V
)random_contrast_1/StatefulPartitionedCall)random_contrast_1/StatefulPartitionedCall2N
%random_flip_1/StatefulPartitionedCall%random_flip_1/StatefulPartitionedCall2V
)random_rotation_1/StatefulPartitionedCall)random_rotation_1/StatefulPartitionedCall2\
,random_translation_1/StatefulPartitionedCall,random_translation_1/StatefulPartitionedCall:j f
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1
_user_specified_namerandom_rotation_1_input
Î

1__inference_random_rotation_1_layer_call_fn_40744

inputs
unknown:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38429y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤/
×
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38098

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maska
stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ì
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: 
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¢
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38037

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_1_layer_call_fn_40504

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38634
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
¼
C
'__inference_re_lu_1_layer_call_fn_40545

inputs
identity¸
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_38805h
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
Å
}
-__inference_random_flip_1_layer_call_fn_41010

inputs
unknown:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38169y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38578

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
ô	
e
F__inference_top_dropout_layer_call_and_return_conditional_losses_40689

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
§

ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_38818

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
Ó
ì
O__inference_random_translation_1_layer_call_and_return_conditional_losses_40998

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip¢!stateful_uniform_1/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mulMulstateful_uniform:z:0Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:[
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½[
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=b
stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_1/ProdProd!stateful_uniform_1/shape:output:0!stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: [
stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :s
stateful_uniform_1/Cast_1Cast stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Þ
!stateful_uniform_1/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource"stateful_uniform_1/Cast/x:output:0stateful_uniform_1/Cast_1:y:0 ^stateful_uniform/RngReadAndSkip*
_output_shapes
:p
&stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
 stateful_uniform_1/strided_sliceStridedSlice)stateful_uniform_1/RngReadAndSkip:value:0/stateful_uniform_1/strided_slice/stack:output:01stateful_uniform_1/strided_slice/stack_1:output:01stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
stateful_uniform_1/BitcastBitcast)stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0r
(stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
"stateful_uniform_1/strided_slice_1StridedSlice)stateful_uniform_1/RngReadAndSkip:value:01stateful_uniform_1/strided_slice_1/stack:output:03stateful_uniform_1/strided_slice_1/stack_1:output:03stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform_1/Bitcast_1Bitcast+stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0q
/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
+stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2!stateful_uniform_1/shape:output:0%stateful_uniform_1/Bitcast_1:output:0#stateful_uniform_1/Bitcast:output:08stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: ¡
stateful_uniform_1/mulMul4stateful_uniform_1/StatelessRandomUniformV2:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniform_1AddV2stateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
mul_1Mulstateful_uniform_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2	mul_1:z:0mul:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
translation_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:p
&translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 translation_matrix/strided_sliceStridedSlice!translation_matrix/Shape:output:0/translation_matrix/strided_slice/stack:output:01translation_matrix/strided_slice/stack_1:output:01translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ª
translation_matrix/ones/packedPack)translation_matrix/strided_slice:output:0)translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:b
translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
translation_matrix/onesFill'translation_matrix/ones/packed:output:0&translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¬
translation_matrix/zeros/packedPack)translation_matrix/strided_slice:output:0*translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
translation_matrix/zerosFill(translation_matrix/zeros/packed:output:0'translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
(translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         î
"translation_matrix/strided_slice_1StridedSliceconcat:output:01translation_matrix/strided_slice_1/stack:output:03translation_matrix/strided_slice_1/stack_1:output:03translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask|
translation_matrix/NegNeg+translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :°
!translation_matrix/zeros_1/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:e
 translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
translation_matrix/zeros_1Fill*translation_matrix/zeros_1/packed:output:0)translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :®
 translation_matrix/ones_1/packedPack)translation_matrix/strided_slice:output:0+translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:d
translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
translation_matrix/ones_1Fill)translation_matrix/ones_1/packed:output:0(translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
(translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
*translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         î
"translation_matrix/strided_slice_2StridedSliceconcat:output:01translation_matrix/strided_slice_2/stack:output:03translation_matrix/strided_slice_2/stack_1:output:03translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask~
translation_matrix/Neg_1Neg+translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :°
!translation_matrix/zeros_2/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:e
 translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
translation_matrix/zeros_2Fill*translation_matrix/zeros_2/packed:output:0)translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ö
translation_matrix/concatConcatV2 translation_matrix/ones:output:0!translation_matrix/zeros:output:0translation_matrix/Neg:y:0#translation_matrix/zeros_1:output:0"translation_matrix/ones_1:output:0translation_matrix/Neg_1:y:0#translation_matrix/zeros_2:output:0'translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¤
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputs"translation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^stateful_uniform/RngReadAndSkip"^stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip2F
!stateful_uniform_1/RngReadAndSkip!stateful_uniform_1/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
M
1__inference_random_contrast_1_layer_call_fn_41077

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38043j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38654

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

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_40459

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

_
C__inference_avg_pool_layer_call_and_return_conditional_losses_40662

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
ÓN

G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39122

inputs$
img_augmentation_39052:	$
img_augmentation_39054:	$
img_augmentation_39056:	$
img_augmentation_39058:	&
conv2d_39061:		 
conv2d_39063: '
batch_normalization_39066: '
batch_normalization_39068: '
batch_normalization_39070: '
batch_normalization_39072: (
conv2d_1_39077: @
conv2d_1_39079:@)
batch_normalization_1_39082:@)
batch_normalization_1_39084:@)
batch_normalization_1_39086:@)
batch_normalization_1_39088:@(
conv2d_2_39093:@@
conv2d_2_39095:@)
batch_normalization_2_39098:@)
batch_normalization_2_39100:@)
batch_normalization_2_39102:@)
batch_normalization_2_39104:@

pred_39110:@

pred_39112:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢(img_augmentation/StatefulPartitionedCall¢pred/StatefulPartitionedCall¢-pred/kernel/Regularizer/Square/ReadVariableOp¢#top_dropout/StatefulPartitionedCallÈ
(img_augmentation/StatefulPartitionedCallStatefulPartitionedCallinputsimg_augmentation_39052img_augmentation_39054img_augmentation_39056img_augmentation_39058*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38457
conv2d/StatefulPartitionedCallStatefulPartitionedCall1img_augmentation/StatefulPartitionedCall:output:0conv2d_39061conv2d_39063*
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
GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_38752ý
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_39066batch_normalization_39068batch_normalization_39070batch_normalization_39072*
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
GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_38558ê
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
@__inference_re_lu_layer_call_and_return_conditional_losses_38772ä
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38578
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_39077conv2d_1_39079*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38785
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_39082batch_normalization_1_39084batch_normalization_1_39086batch_normalization_1_39088*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38634ð
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_38805ê
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38654
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_39093conv2d_2_39095*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_38818
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_39098batch_normalization_2_39100batch_normalization_2_39102batch_normalization_2_39104*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38710ð
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_38838Ô
avg_pool/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
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
GPU2*0J 8 *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_38731ë
#top_dropout/StatefulPartitionedCallStatefulPartitionedCall!avg_pool/PartitionedCall:output:0*
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
F__inference_top_dropout_layer_call_and_return_conditional_losses_38951
pred/StatefulPartitionedCallStatefulPartitionedCall,top_dropout/StatefulPartitionedCall:output:0
pred_39110
pred_39112*
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
GPU2*0J 8 *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_38865x
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
pred_39110*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^img_augmentation/StatefulPartitionedCall^pred/StatefulPartitionedCall.^pred/kernel/Regularizer/Square/ReadVariableOp$^top_dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(img_augmentation/StatefulPartitionedCall(img_augmentation/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp2J
#top_dropout/StatefulPartitionedCall#top_dropout/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
I
-__inference_random_flip_1_layer_call_fn_41003

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38037j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
M
1__inference_random_rotation_1_layer_call_fn_40737

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38025j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40579

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

h
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_41088

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_40651

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
ô	
e
F__inference_top_dropout_layer_call_and_return_conditional_losses_38951

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
	
Î
3__inference_batch_normalization_layer_call_fn_40390

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_38527
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
ë

(__inference_conv2d_1_layer_call_fn_40468

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallã
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38785w
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
ð
 
?__inference_pred_layer_call_and_return_conditional_losses_40721

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢-pred/kernel/Regularizer/Square/ReadVariableOpt
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
:ÿÿÿÿÿÿÿÿÿ
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^pred/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ö
L
0__inference_img_augmentation_layer_call_fn_40020

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38046j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38679

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
î

¨
__inference_loss_fn_0_40732H
6pred_kernel_regularizer_square_readvariableop_resource:@
identity¢-pred/kernel/Regularizer/Square/ReadVariableOp¤
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6pred_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentitypred/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: v
NoOpNoOp.^pred/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp
¤/
×
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_41129

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maska
stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ì
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: 
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¢
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

(__inference_conv2d_2_layer_call_fn_40569

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallã
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_38818w
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
×
½
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40439

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
§

ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38785

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
Ù
¿
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_40540

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
É

N__inference_batch_normalization_layer_call_and_return_conditional_losses_38527

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

d
H__inference_random_flip_1_layer_call_and_return_conditional_losses_41014

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
D
(__inference_avg_pool_layer_call_fn_40656

inputs
identityº
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
GPU2*0J 8 *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_38731i
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
ÝM
Ó
H__inference_random_flip_1_layer_call_and_return_conditional_losses_41072

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ë
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
A
%__inference_re_lu_layer_call_fn_40444

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
@__inference_re_lu_layer_call_and_return_conditional_losses_38772h
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
ð
 
?__inference_pred_layer_call_and_return_conditional_losses_38865

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢-pred/kernel/Regularizer/Square/ReadVariableOpt
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
:ÿÿÿÿÿÿÿÿÿ
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^pred/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38603

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

g
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38046

inputs
identityÖ
!random_rotation_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38025
$random_translation_1/PartitionedCallPartitionedCall*random_rotation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38031õ
random_flip_1/PartitionedCallPartitionedCall-random_translation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38037ö
!random_contrast_1/PartitionedCallPartitionedCall&random_flip_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38043|
IdentityIdentity*random_contrast_1/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

´
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38457

inputs%
random_rotation_1_38444:	(
random_translation_1_38447:	!
random_flip_1_38450:	%
random_contrast_1_38453:	
identity¢)random_contrast_1/StatefulPartitionedCall¢%random_flip_1/StatefulPartitionedCall¢)random_rotation_1/StatefulPartitionedCall¢,random_translation_1/StatefulPartitionedCall
)random_rotation_1/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_rotation_1_38444*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38429µ
,random_translation_1/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_1/StatefulPartitionedCall:output:0random_translation_1_38447*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38298£
%random_flip_1/StatefulPartitionedCallStatefulPartitionedCall5random_translation_1/StatefulPartitionedCall:output:0random_flip_1_38450*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38169¨
)random_contrast_1/StatefulPartitionedCallStatefulPartitionedCall.random_flip_1/StatefulPartitionedCall:output:0random_contrast_1_38453*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_38098
IdentityIdentity2random_contrast_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
NoOpNoOp*^random_contrast_1/StatefulPartitionedCall&^random_flip_1/StatefulPartitionedCall*^random_rotation_1/StatefulPartitionedCall-^random_translation_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : 2V
)random_contrast_1/StatefulPartitionedCall)random_contrast_1/StatefulPartitionedCall2N
%random_flip_1/StatefulPartitionedCall%random_flip_1/StatefulPartitionedCall2V
)random_rotation_1/StatefulPartitionedCall)random_rotation_1/StatefulPartitionedCall2\
,random_translation_1/StatefulPartitionedCall,random_translation_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

]
0__inference_img_augmentation_layer_call_fn_38049
random_rotation_1_input
identityÔ
PartitionedCallPartitionedCallrandom_rotation_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38046j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:j f
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1
_user_specified_namerandom_rotation_1_input
äI
³	
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39291
input_1&
conv2d_39230:		 
conv2d_39232: '
batch_normalization_39235: '
batch_normalization_39237: '
batch_normalization_39239: '
batch_normalization_39241: (
conv2d_1_39246: @
conv2d_1_39248:@)
batch_normalization_1_39251:@)
batch_normalization_1_39253:@)
batch_normalization_1_39255:@)
batch_normalization_1_39257:@(
conv2d_2_39262:@@
conv2d_2_39264:@)
batch_normalization_2_39267:@)
batch_normalization_2_39269:@)
batch_normalization_2_39271:@)
batch_normalization_2_39273:@

pred_39279:@

pred_39281:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢pred/StatefulPartitionedCall¢-pred/kernel/Regularizer/Square/ReadVariableOpÕ
 img_augmentation/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38046
conv2d/StatefulPartitionedCallStatefulPartitionedCall)img_augmentation/PartitionedCall:output:0conv2d_39230conv2d_39232*
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
GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_38752ÿ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_39235batch_normalization_39237batch_normalization_39239batch_normalization_39241*
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
GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_38527ê
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
@__inference_re_lu_layer_call_and_return_conditional_losses_38772ä
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
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
GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38578
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_39246conv2d_1_39248*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_38785
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_39251batch_normalization_1_39253batch_normalization_1_39255batch_normalization_1_39257*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38603ð
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_38805ê
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
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
GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38654
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_39262conv2d_2_39264*
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
GPU2*0J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_38818
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_39267batch_normalization_2_39269batch_normalization_2_39271batch_normalization_2_39273*
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38679ð
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_38838Ô
avg_pool/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
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
GPU2*0J 8 *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_38731Û
top_dropout/PartitionedCallPartitionedCall!avg_pool/PartitionedCall:output:0*
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
F__inference_top_dropout_layer_call_and_return_conditional_losses_38846þ
pred/StatefulPartitionedCallStatefulPartitionedCall$top_dropout/PartitionedCall:output:0
pred_39279
pred_39281*
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
GPU2*0J 8 *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_38865x
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
pred_39279*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^pred/StatefulPartitionedCall.^pred/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
	
Ð
5__inference_batch_normalization_1_layer_call_fn_40491

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38603
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

k
O__inference_random_translation_1_layer_call_and_return_conditional_losses_40882

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¿
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_38634

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
´e
ò
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39559

inputs?
%conv2d_conv2d_readvariableop_resource:		 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@5
#pred_matmul_readvariableop_resource:@2
$pred_biasadd_readvariableop_resource:
identity¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢pred/BiasAdd/ReadVariableOp¢pred/MatMul/ReadVariableOp¢-pred/kernel/Regularizer/Square/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0¨
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? 
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0«
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ?? : : : : :*
epsilon%o:*
is_training( v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? §
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0·
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ		@:@:@:@:@:*
epsilon%o:*
is_training( z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@«
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Æ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0·
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( z
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
avg_pool/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
avg_pool/MaxMaxre_lu_2/Relu:activations:0'avg_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
top_dropout/IdentityIdentityavg_pool/Max:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
pred/MatMul/ReadVariableOpReadVariableOp#pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
pred/MatMulMatMultop_dropout/Identity:output:0"pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
pred/BiasAdd/ReadVariableOpReadVariableOp$pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
pred/BiasAddBiasAddpred/MatMul:product:0#pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
pred/SoftmaxSoftmaxpred/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitypred/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^pred/BiasAdd/ReadVariableOp^pred/MatMul/ReadVariableOp.^pred/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2:
pred/BiasAdd/ReadVariableOppred/BiasAdd/ReadVariableOp28
pred/MatMul/ReadVariableOppred/MatMul/ReadVariableOp2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

k
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38031

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
d
F__inference_top_dropout_layer_call_and_return_conditional_losses_40677

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
æ
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_40550

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
Ó
ì
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38298

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip¢!stateful_uniform_1/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mulMulstateful_uniform:z:0Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:[
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½[
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=b
stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_1/ProdProd!stateful_uniform_1/shape:output:0!stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: [
stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :s
stateful_uniform_1/Cast_1Cast stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Þ
!stateful_uniform_1/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource"stateful_uniform_1/Cast/x:output:0stateful_uniform_1/Cast_1:y:0 ^stateful_uniform/RngReadAndSkip*
_output_shapes
:p
&stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
 stateful_uniform_1/strided_sliceStridedSlice)stateful_uniform_1/RngReadAndSkip:value:0/stateful_uniform_1/strided_slice/stack:output:01stateful_uniform_1/strided_slice/stack_1:output:01stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
stateful_uniform_1/BitcastBitcast)stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0r
(stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
"stateful_uniform_1/strided_slice_1StridedSlice)stateful_uniform_1/RngReadAndSkip:value:01stateful_uniform_1/strided_slice_1/stack:output:03stateful_uniform_1/strided_slice_1/stack_1:output:03stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform_1/Bitcast_1Bitcast+stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0q
/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
+stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2!stateful_uniform_1/shape:output:0%stateful_uniform_1/Bitcast_1:output:0#stateful_uniform_1/Bitcast:output:08stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: ¡
stateful_uniform_1/mulMul4stateful_uniform_1/StatelessRandomUniformV2:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniform_1AddV2stateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
mul_1Mulstateful_uniform_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2	mul_1:z:0mul:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
translation_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:p
&translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 translation_matrix/strided_sliceStridedSlice!translation_matrix/Shape:output:0/translation_matrix/strided_slice/stack:output:01translation_matrix/strided_slice/stack_1:output:01translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ª
translation_matrix/ones/packedPack)translation_matrix/strided_slice:output:0)translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:b
translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
translation_matrix/onesFill'translation_matrix/ones/packed:output:0&translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¬
translation_matrix/zeros/packedPack)translation_matrix/strided_slice:output:0*translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
translation_matrix/zerosFill(translation_matrix/zeros/packed:output:0'translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
(translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         î
"translation_matrix/strided_slice_1StridedSliceconcat:output:01translation_matrix/strided_slice_1/stack:output:03translation_matrix/strided_slice_1/stack_1:output:03translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask|
translation_matrix/NegNeg+translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :°
!translation_matrix/zeros_1/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:e
 translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
translation_matrix/zeros_1Fill*translation_matrix/zeros_1/packed:output:0)translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :®
 translation_matrix/ones_1/packedPack)translation_matrix/strided_slice:output:0+translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:d
translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
translation_matrix/ones_1Fill)translation_matrix/ones_1/packed:output:0(translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
(translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
*translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         î
"translation_matrix/strided_slice_2StridedSliceconcat:output:01translation_matrix/strided_slice_2/stack:output:03translation_matrix/strided_slice_2/stack_1:output:03translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask~
translation_matrix/Neg_1Neg+translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :°
!translation_matrix/zeros_2/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:e
 translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
translation_matrix/zeros_2Fill*translation_matrix/zeros_2/packed:output:0)translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ö
translation_matrix/concatConcatV2 translation_matrix/ones:output:0!translation_matrix/zeros:output:0translation_matrix/Neg:y:0#translation_matrix/zeros_1:output:0"translation_matrix/ones_1:output:0translation_matrix/Neg_1:y:0#translation_matrix/zeros_2:output:0'translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¤
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputs"translation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^stateful_uniform/RngReadAndSkip"^stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip2F
!stateful_uniform_1/RngReadAndSkip!stateful_uniform_1/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40866

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq¿Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¡
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_38838

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
ö
'
!__inference__traced_restore_41548
file_prefix8
assignvariableop_conv2d_kernel:		 ,
assignvariableop_1_conv2d_bias: :
,assignvariableop_2_batch_normalization_gamma: 9
+assignvariableop_3_batch_normalization_beta: @
2assignvariableop_4_batch_normalization_moving_mean: D
6assignvariableop_5_batch_normalization_moving_variance: <
"assignvariableop_6_conv2d_1_kernel: @.
 assignvariableop_7_conv2d_1_bias:@<
.assignvariableop_8_batch_normalization_1_gamma:@;
-assignvariableop_9_batch_normalization_1_beta:@C
5assignvariableop_10_batch_normalization_1_moving_mean:@G
9assignvariableop_11_batch_normalization_1_moving_variance:@=
#assignvariableop_12_conv2d_2_kernel:@@/
!assignvariableop_13_conv2d_2_bias:@=
/assignvariableop_14_batch_normalization_2_gamma:@<
.assignvariableop_15_batch_normalization_2_beta:@C
5assignvariableop_16_batch_normalization_2_moving_mean:@G
9assignvariableop_17_batch_normalization_2_moving_variance:@1
assignvariableop_18_pred_kernel:@+
assignvariableop_19_pred_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: #
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: <
.assignvariableop_29_random_rotation_1_statevar:	?
1assignvariableop_30_random_translation_1_statevar:	8
*assignvariableop_31_random_flip_1_statevar:	<
.assignvariableop_32_random_contrast_1_statevar:	B
(assignvariableop_33_adam_conv2d_kernel_m:		 4
&assignvariableop_34_adam_conv2d_bias_m: B
4assignvariableop_35_adam_batch_normalization_gamma_m: A
3assignvariableop_36_adam_batch_normalization_beta_m: D
*assignvariableop_37_adam_conv2d_1_kernel_m: @6
(assignvariableop_38_adam_conv2d_1_bias_m:@D
6assignvariableop_39_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_40_adam_batch_normalization_1_beta_m:@D
*assignvariableop_41_adam_conv2d_2_kernel_m:@@6
(assignvariableop_42_adam_conv2d_2_bias_m:@D
6assignvariableop_43_adam_batch_normalization_2_gamma_m:@C
5assignvariableop_44_adam_batch_normalization_2_beta_m:@8
&assignvariableop_45_adam_pred_kernel_m:@2
$assignvariableop_46_adam_pred_bias_m:B
(assignvariableop_47_adam_conv2d_kernel_v:		 4
&assignvariableop_48_adam_conv2d_bias_v: B
4assignvariableop_49_adam_batch_normalization_gamma_v: A
3assignvariableop_50_adam_batch_normalization_beta_v: D
*assignvariableop_51_adam_conv2d_1_kernel_v: @6
(assignvariableop_52_adam_conv2d_1_bias_v:@D
6assignvariableop_53_adam_batch_normalization_1_gamma_v:@C
5assignvariableop_54_adam_batch_normalization_1_beta_v:@D
*assignvariableop_55_adam_conv2d_2_kernel_v:@@6
(assignvariableop_56_adam_conv2d_2_bias_v:@D
6assignvariableop_57_adam_batch_normalization_2_gamma_v:@C
5assignvariableop_58_adam_batch_normalization_2_beta_v:@8
&assignvariableop_59_adam_pred_kernel_v:@2
$assignvariableop_60_adam_pred_bias_v:
identity_62¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¡"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*Ç!
value½!Bº!>B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-3/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHï
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ×
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesû
ø::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*L
dtypesB
@2>					[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_pred_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_pred_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_29AssignVariableOp.assignvariableop_29_random_rotation_1_statevarIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:¢
AssignVariableOp_30AssignVariableOp1assignvariableop_30_random_translation_1_statevarIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_random_flip_1_statevarIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_32AssignVariableOp.assignvariableop_32_random_contrast_1_statevarIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_conv2d_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_batch_normalization_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_batch_normalization_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_1_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_1_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_2_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_2_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_pred_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp$assignvariableop_46_adam_pred_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv2d_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_conv2d_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_batch_normalization_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_batch_normalization_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_1_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_1_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_1_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_1_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_2_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_2_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_pred_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp$assignvariableop_60_adam_pred_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_62IdentityIdentity_61:output:0^NoOp_1*
T0*
_output_shapes
: ú

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_62Identity_62:output:0*
_input_shapes~
|: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ë

&__inference_conv2d_layer_call_fn_40367

inputs!
unknown:		 
	unknown_0: 
identity¢StatefulPartitionedCallá
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
GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_38752w
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

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40560

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
×
½
N__inference_batch_normalization_layer_call_and_return_conditional_losses_38558

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

_
C__inference_avg_pool_layer_call_and_return_conditional_losses_38731

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
ÝM
Ó
H__inference_random_flip_1_layer_call_and_return_conditional_losses_38169

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ë
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ïµ
Ê
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39968

inputsY
Kimg_augmentation_random_rotation_1_stateful_uniform_rngreadandskip_resource:	\
Nimg_augmentation_random_translation_1_stateful_uniform_rngreadandskip_resource:	^
Pimg_augmentation_random_flip_1_stateful_uniform_full_int_rngreadandskip_resource:	b
Timg_augmentation_random_contrast_1_stateful_uniform_full_int_rngreadandskip_resource:	?
%conv2d_conv2d_readvariableop_resource:		 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@5
#pred_matmul_readvariableop_resource:@2
$pred_biasadd_readvariableop_resource:
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢Kimg_augmentation/random_contrast_1/stateful_uniform_full_int/RngReadAndSkip¢Gimg_augmentation/random_flip_1/stateful_uniform_full_int/RngReadAndSkip¢Bimg_augmentation/random_rotation_1/stateful_uniform/RngReadAndSkip¢Eimg_augmentation/random_translation_1/stateful_uniform/RngReadAndSkip¢Gimg_augmentation/random_translation_1/stateful_uniform_1/RngReadAndSkip¢pred/BiasAdd/ReadVariableOp¢pred/MatMul/ReadVariableOp¢-pred/kernel/Regularizer/Square/ReadVariableOp^
(img_augmentation/random_rotation_1/ShapeShapeinputs*
T0*
_output_shapes
:
6img_augmentation/random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8img_augmentation/random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8img_augmentation/random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0img_augmentation/random_rotation_1/strided_sliceStridedSlice1img_augmentation/random_rotation_1/Shape:output:0?img_augmentation/random_rotation_1/strided_slice/stack:output:0Aimg_augmentation/random_rotation_1/strided_slice/stack_1:output:0Aimg_augmentation/random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
8img_augmentation/random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2img_augmentation/random_rotation_1/strided_slice_1StridedSlice1img_augmentation/random_rotation_1/Shape:output:0Aimg_augmentation/random_rotation_1/strided_slice_1/stack:output:0Cimg_augmentation/random_rotation_1/strided_slice_1/stack_1:output:0Cimg_augmentation/random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'img_augmentation/random_rotation_1/CastCast;img_augmentation/random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
8img_augmentation/random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2img_augmentation/random_rotation_1/strided_slice_2StridedSlice1img_augmentation/random_rotation_1/Shape:output:0Aimg_augmentation/random_rotation_1/strided_slice_2/stack:output:0Cimg_augmentation/random_rotation_1/strided_slice_2/stack_1:output:0Cimg_augmentation/random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
)img_augmentation/random_rotation_1/Cast_1Cast;img_augmentation/random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: ª
9img_augmentation/random_rotation_1/stateful_uniform/shapePack9img_augmentation/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:|
7img_augmentation/random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq¿|
7img_augmentation/random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?
9img_augmentation/random_rotation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
8img_augmentation/random_rotation_1/stateful_uniform/ProdProdBimg_augmentation/random_rotation_1/stateful_uniform/shape:output:0Bimg_augmentation/random_rotation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: |
:img_augmentation/random_rotation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :µ
:img_augmentation/random_rotation_1/stateful_uniform/Cast_1CastAimg_augmentation/random_rotation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Â
Bimg_augmentation/random_rotation_1/stateful_uniform/RngReadAndSkipRngReadAndSkipKimg_augmentation_random_rotation_1_stateful_uniform_rngreadandskip_resourceCimg_augmentation/random_rotation_1/stateful_uniform/Cast/x:output:0>img_augmentation/random_rotation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Gimg_augmentation/random_rotation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Iimg_augmentation/random_rotation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Iimg_augmentation/random_rotation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
Aimg_augmentation/random_rotation_1/stateful_uniform/strided_sliceStridedSliceJimg_augmentation/random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Pimg_augmentation/random_rotation_1/stateful_uniform/strided_slice/stack:output:0Rimg_augmentation/random_rotation_1/stateful_uniform/strided_slice/stack_1:output:0Rimg_augmentation/random_rotation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÃ
;img_augmentation/random_rotation_1/stateful_uniform/BitcastBitcastJimg_augmentation/random_rotation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Iimg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Kimg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Kimg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
Cimg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1StridedSliceJimg_augmentation/random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Rimg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1/stack:output:0Timg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1/stack_1:output:0Timg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ç
=img_augmentation/random_rotation_1/stateful_uniform/Bitcast_1BitcastLimg_augmentation/random_rotation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Pimg_augmentation/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :º
Limg_augmentation/random_rotation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Bimg_augmentation/random_rotation_1/stateful_uniform/shape:output:0Fimg_augmentation/random_rotation_1/stateful_uniform/Bitcast_1:output:0Dimg_augmentation/random_rotation_1/stateful_uniform/Bitcast:output:0Yimg_augmentation/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
7img_augmentation/random_rotation_1/stateful_uniform/subSub@img_augmentation/random_rotation_1/stateful_uniform/max:output:0@img_augmentation/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 
7img_augmentation/random_rotation_1/stateful_uniform/mulMulUimg_augmentation/random_rotation_1/stateful_uniform/StatelessRandomUniformV2:output:0;img_augmentation/random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
3img_augmentation/random_rotation_1/stateful_uniformAddV2;img_augmentation/random_rotation_1/stateful_uniform/mul:z:0@img_augmentation/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
8img_augmentation/random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ð
6img_augmentation/random_rotation_1/rotation_matrix/subSub-img_augmentation/random_rotation_1/Cast_1:y:0Aimg_augmentation/random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ¤
6img_augmentation/random_rotation_1/rotation_matrix/CosCos7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ô
8img_augmentation/random_rotation_1/rotation_matrix/sub_1Sub-img_augmentation/random_rotation_1/Cast_1:y:0Cimg_augmentation/random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: å
6img_augmentation/random_rotation_1/rotation_matrix/mulMul:img_augmentation/random_rotation_1/rotation_matrix/Cos:y:0<img_augmentation/random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
6img_augmentation/random_rotation_1/rotation_matrix/SinSin7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ò
8img_augmentation/random_rotation_1/rotation_matrix/sub_2Sub+img_augmentation/random_rotation_1/Cast:y:0Cimg_augmentation/random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ç
8img_augmentation/random_rotation_1/rotation_matrix/mul_1Mul:img_augmentation/random_rotation_1/rotation_matrix/Sin:y:0<img_augmentation/random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
8img_augmentation/random_rotation_1/rotation_matrix/sub_3Sub:img_augmentation/random_rotation_1/rotation_matrix/mul:z:0<img_augmentation/random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
8img_augmentation/random_rotation_1/rotation_matrix/sub_4Sub:img_augmentation/random_rotation_1/rotation_matrix/sub:z:0<img_augmentation/random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<img_augmentation/random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ø
:img_augmentation/random_rotation_1/rotation_matrix/truedivRealDiv<img_augmentation/random_rotation_1/rotation_matrix/sub_4:z:0Eimg_augmentation/random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ò
8img_augmentation/random_rotation_1/rotation_matrix/sub_5Sub+img_augmentation/random_rotation_1/Cast:y:0Cimg_augmentation/random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: ¦
8img_augmentation/random_rotation_1/rotation_matrix/Sin_1Sin7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ô
8img_augmentation/random_rotation_1/rotation_matrix/sub_6Sub-img_augmentation/random_rotation_1/Cast_1:y:0Cimg_augmentation/random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: é
8img_augmentation/random_rotation_1/rotation_matrix/mul_2Mul<img_augmentation/random_rotation_1/rotation_matrix/Sin_1:y:0<img_augmentation/random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
8img_augmentation/random_rotation_1/rotation_matrix/Cos_1Cos7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:img_augmentation/random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ò
8img_augmentation/random_rotation_1/rotation_matrix/sub_7Sub+img_augmentation/random_rotation_1/Cast:y:0Cimg_augmentation/random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: é
8img_augmentation/random_rotation_1/rotation_matrix/mul_3Mul<img_augmentation/random_rotation_1/rotation_matrix/Cos_1:y:0<img_augmentation/random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
6img_augmentation/random_rotation_1/rotation_matrix/addAddV2<img_augmentation/random_rotation_1/rotation_matrix/mul_2:z:0<img_augmentation/random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
8img_augmentation/random_rotation_1/rotation_matrix/sub_8Sub<img_augmentation/random_rotation_1/rotation_matrix/sub_5:z:0:img_augmentation/random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>img_augmentation/random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ü
<img_augmentation/random_rotation_1/rotation_matrix/truediv_1RealDiv<img_augmentation/random_rotation_1/rotation_matrix/sub_8:z:0Gimg_augmentation/random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8img_augmentation/random_rotation_1/rotation_matrix/ShapeShape7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:
Fimg_augmentation/random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ð
@img_augmentation/random_rotation_1/rotation_matrix/strided_sliceStridedSliceAimg_augmentation/random_rotation_1/rotation_matrix/Shape:output:0Oimg_augmentation/random_rotation_1/rotation_matrix/strided_slice/stack:output:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¦
8img_augmentation/random_rotation_1/rotation_matrix/Cos_2Cos7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Bimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1StridedSlice<img_augmentation/random_rotation_1/rotation_matrix/Cos_2:y:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¦
8img_augmentation/random_rotation_1/rotation_matrix/Sin_2Sin7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Bimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2StridedSlice<img_augmentation/random_rotation_1/rotation_matrix/Sin_2:y:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¼
6img_augmentation/random_rotation_1/rotation_matrix/NegNegKimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Bimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3StridedSlice>img_augmentation/random_rotation_1/rotation_matrix/truediv:z:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¦
8img_augmentation/random_rotation_1/rotation_matrix/Sin_3Sin7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Bimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4StridedSlice<img_augmentation/random_rotation_1/rotation_matrix/Sin_3:y:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¦
8img_augmentation/random_rotation_1/rotation_matrix/Cos_3Cos7img_augmentation/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Bimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5StridedSlice<img_augmentation/random_rotation_1/rotation_matrix/Cos_3:y:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
Himg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Jimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Bimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6StridedSlice@img_augmentation/random_rotation_1/rotation_matrix/truediv_1:z:0Qimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Simg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
Aimg_augmentation/random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
?img_augmentation/random_rotation_1/rotation_matrix/zeros/packedPackIimg_augmentation/random_rotation_1/rotation_matrix/strided_slice:output:0Jimg_augmentation/random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
>img_augmentation/random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
8img_augmentation/random_rotation_1/rotation_matrix/zerosFillHimg_augmentation/random_rotation_1/rotation_matrix/zeros/packed:output:0Gimg_augmentation/random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>img_augmentation/random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :É
9img_augmentation/random_rotation_1/rotation_matrix/concatConcatV2Kimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_1:output:0:img_augmentation/random_rotation_1/rotation_matrix/Neg:y:0Kimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_3:output:0Kimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_4:output:0Kimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_5:output:0Kimg_augmentation/random_rotation_1/rotation_matrix/strided_slice_6:output:0Aimg_augmentation/random_rotation_1/rotation_matrix/zeros:output:0Gimg_augmentation/random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
2img_augmentation/random_rotation_1/transform/ShapeShapeinputs*
T0*
_output_shapes
:
@img_augmentation/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Bimg_augmentation/random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Bimg_augmentation/random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
:img_augmentation/random_rotation_1/transform/strided_sliceStridedSlice;img_augmentation/random_rotation_1/transform/Shape:output:0Iimg_augmentation/random_rotation_1/transform/strided_slice/stack:output:0Kimg_augmentation/random_rotation_1/transform/strided_slice/stack_1:output:0Kimg_augmentation/random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:|
7img_augmentation/random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ­
Gimg_augmentation/random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsBimg_augmentation/random_rotation_1/rotation_matrix/concat:output:0Cimg_augmentation/random_rotation_1/transform/strided_slice:output:0@img_augmentation/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR·
+img_augmentation/random_translation_1/ShapeShape\img_augmentation/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:
9img_augmentation/random_translation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;img_augmentation/random_translation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;img_augmentation/random_translation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3img_augmentation/random_translation_1/strided_sliceStridedSlice4img_augmentation/random_translation_1/Shape:output:0Bimg_augmentation/random_translation_1/strided_slice/stack:output:0Dimg_augmentation/random_translation_1/strided_slice/stack_1:output:0Dimg_augmentation/random_translation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
;img_augmentation/random_translation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
=img_augmentation/random_translation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=img_augmentation/random_translation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5img_augmentation/random_translation_1/strided_slice_1StridedSlice4img_augmentation/random_translation_1/Shape:output:0Dimg_augmentation/random_translation_1/strided_slice_1/stack:output:0Fimg_augmentation/random_translation_1/strided_slice_1/stack_1:output:0Fimg_augmentation/random_translation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¢
*img_augmentation/random_translation_1/CastCast>img_augmentation/random_translation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
;img_augmentation/random_translation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=img_augmentation/random_translation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
=img_augmentation/random_translation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5img_augmentation/random_translation_1/strided_slice_2StridedSlice4img_augmentation/random_translation_1/Shape:output:0Dimg_augmentation/random_translation_1/strided_slice_2/stack:output:0Fimg_augmentation/random_translation_1/strided_slice_2/stack_1:output:0Fimg_augmentation/random_translation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¤
,img_augmentation/random_translation_1/Cast_1Cast>img_augmentation/random_translation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 
>img_augmentation/random_translation_1/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ù
<img_augmentation/random_translation_1/stateful_uniform/shapePack<img_augmentation/random_translation_1/strided_slice:output:0Gimg_augmentation/random_translation_1/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:
:img_augmentation/random_translation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½
:img_augmentation/random_translation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
<img_augmentation/random_translation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ò
;img_augmentation/random_translation_1/stateful_uniform/ProdProdEimg_augmentation/random_translation_1/stateful_uniform/shape:output:0Eimg_augmentation/random_translation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 
=img_augmentation/random_translation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :»
=img_augmentation/random_translation_1/stateful_uniform/Cast_1CastDimg_augmentation/random_translation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Î
Eimg_augmentation/random_translation_1/stateful_uniform/RngReadAndSkipRngReadAndSkipNimg_augmentation_random_translation_1_stateful_uniform_rngreadandskip_resourceFimg_augmentation/random_translation_1/stateful_uniform/Cast/x:output:0Aimg_augmentation/random_translation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Jimg_augmentation/random_translation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Limg_augmentation/random_translation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Limg_augmentation/random_translation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
Dimg_augmentation/random_translation_1/stateful_uniform/strided_sliceStridedSliceMimg_augmentation/random_translation_1/stateful_uniform/RngReadAndSkip:value:0Simg_augmentation/random_translation_1/stateful_uniform/strided_slice/stack:output:0Uimg_augmentation/random_translation_1/stateful_uniform/strided_slice/stack_1:output:0Uimg_augmentation/random_translation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÉ
>img_augmentation/random_translation_1/stateful_uniform/BitcastBitcastMimg_augmentation/random_translation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Limg_augmentation/random_translation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Nimg_augmentation/random_translation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nimg_augmentation/random_translation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
Fimg_augmentation/random_translation_1/stateful_uniform/strided_slice_1StridedSliceMimg_augmentation/random_translation_1/stateful_uniform/RngReadAndSkip:value:0Uimg_augmentation/random_translation_1/stateful_uniform/strided_slice_1/stack:output:0Wimg_augmentation/random_translation_1/stateful_uniform/strided_slice_1/stack_1:output:0Wimg_augmentation/random_translation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Í
@img_augmentation/random_translation_1/stateful_uniform/Bitcast_1BitcastOimg_augmentation/random_translation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Simg_augmentation/random_translation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Í
Oimg_augmentation/random_translation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Eimg_augmentation/random_translation_1/stateful_uniform/shape:output:0Iimg_augmentation/random_translation_1/stateful_uniform/Bitcast_1:output:0Gimg_augmentation/random_translation_1/stateful_uniform/Bitcast:output:0\img_augmentation/random_translation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
:img_augmentation/random_translation_1/stateful_uniform/subSubCimg_augmentation/random_translation_1/stateful_uniform/max:output:0Cimg_augmentation/random_translation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 
:img_augmentation/random_translation_1/stateful_uniform/mulMulXimg_augmentation/random_translation_1/stateful_uniform/StatelessRandomUniformV2:output:0>img_augmentation/random_translation_1/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
6img_augmentation/random_translation_1/stateful_uniformAddV2>img_augmentation/random_translation_1/stateful_uniform/mul:z:0Cimg_augmentation/random_translation_1/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
)img_augmentation/random_translation_1/mulMul:img_augmentation/random_translation_1/stateful_uniform:z:0.img_augmentation/random_translation_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@img_augmentation/random_translation_1/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ý
>img_augmentation/random_translation_1/stateful_uniform_1/shapePack<img_augmentation/random_translation_1/strided_slice:output:0Iimg_augmentation/random_translation_1/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:
<img_augmentation/random_translation_1/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½
<img_augmentation/random_translation_1/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
>img_augmentation/random_translation_1/stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ø
=img_augmentation/random_translation_1/stateful_uniform_1/ProdProdGimg_augmentation/random_translation_1/stateful_uniform_1/shape:output:0Gimg_augmentation/random_translation_1/stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: 
?img_augmentation/random_translation_1/stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¿
?img_augmentation/random_translation_1/stateful_uniform_1/Cast_1CastFimg_augmentation/random_translation_1/stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
Gimg_augmentation/random_translation_1/stateful_uniform_1/RngReadAndSkipRngReadAndSkipNimg_augmentation_random_translation_1_stateful_uniform_rngreadandskip_resourceHimg_augmentation/random_translation_1/stateful_uniform_1/Cast/x:output:0Cimg_augmentation/random_translation_1/stateful_uniform_1/Cast_1:y:0F^img_augmentation/random_translation_1/stateful_uniform/RngReadAndSkip*
_output_shapes
:
Limg_augmentation/random_translation_1/stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
Fimg_augmentation/random_translation_1/stateful_uniform_1/strided_sliceStridedSliceOimg_augmentation/random_translation_1/stateful_uniform_1/RngReadAndSkip:value:0Uimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice/stack:output:0Wimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice/stack_1:output:0Wimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÍ
@img_augmentation/random_translation_1/stateful_uniform_1/BitcastBitcastOimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Nimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Pimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Pimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
Himg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1StridedSliceOimg_augmentation/random_translation_1/stateful_uniform_1/RngReadAndSkip:value:0Wimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1/stack:output:0Yimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1/stack_1:output:0Yimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ñ
Bimg_augmentation/random_translation_1/stateful_uniform_1/Bitcast_1BitcastQimg_augmentation/random_translation_1/stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Uimg_augmentation/random_translation_1/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :×
Qimg_augmentation/random_translation_1/stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2Gimg_augmentation/random_translation_1/stateful_uniform_1/shape:output:0Kimg_augmentation/random_translation_1/stateful_uniform_1/Bitcast_1:output:0Iimg_augmentation/random_translation_1/stateful_uniform_1/Bitcast:output:0^img_augmentation/random_translation_1/stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
<img_augmentation/random_translation_1/stateful_uniform_1/subSubEimg_augmentation/random_translation_1/stateful_uniform_1/max:output:0Eimg_augmentation/random_translation_1/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 
<img_augmentation/random_translation_1/stateful_uniform_1/mulMulZimg_augmentation/random_translation_1/stateful_uniform_1/StatelessRandomUniformV2:output:0@img_augmentation/random_translation_1/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
8img_augmentation/random_translation_1/stateful_uniform_1AddV2@img_augmentation/random_translation_1/stateful_uniform_1/mul:z:0Eimg_augmentation/random_translation_1/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
+img_augmentation/random_translation_1/mul_1Mul<img_augmentation/random_translation_1/stateful_uniform_1:z:00img_augmentation/random_translation_1/Cast_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1img_augmentation/random_translation_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
,img_augmentation/random_translation_1/concatConcatV2/img_augmentation/random_translation_1/mul_1:z:0-img_augmentation/random_translation_1/mul:z:0:img_augmentation/random_translation_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
>img_augmentation/random_translation_1/translation_matrix/ShapeShape5img_augmentation/random_translation_1/concat:output:0*
T0*
_output_shapes
:
Limg_augmentation/random_translation_1/translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nimg_augmentation/random_translation_1/translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nimg_augmentation/random_translation_1/translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:î
Fimg_augmentation/random_translation_1/translation_matrix/strided_sliceStridedSliceGimg_augmentation/random_translation_1/translation_matrix/Shape:output:0Uimg_augmentation/random_translation_1/translation_matrix/strided_slice/stack:output:0Wimg_augmentation/random_translation_1/translation_matrix/strided_slice/stack_1:output:0Wimg_augmentation/random_translation_1/translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Fimg_augmentation/random_translation_1/translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Dimg_augmentation/random_translation_1/translation_matrix/ones/packedPackOimg_augmentation/random_translation_1/translation_matrix/strided_slice:output:0Oimg_augmentation/random_translation_1/translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:
Cimg_augmentation/random_translation_1/translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
=img_augmentation/random_translation_1/translation_matrix/onesFillMimg_augmentation/random_translation_1/translation_matrix/ones/packed:output:0Limg_augmentation/random_translation_1/translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Gimg_augmentation/random_translation_1/translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Eimg_augmentation/random_translation_1/translation_matrix/zeros/packedPackOimg_augmentation/random_translation_1/translation_matrix/strided_slice:output:0Pimg_augmentation/random_translation_1/translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
Dimg_augmentation/random_translation_1/translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
>img_augmentation/random_translation_1/translation_matrix/zerosFillNimg_augmentation/random_translation_1/translation_matrix/zeros/packed:output:0Mimg_augmentation/random_translation_1/translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
Nimg_augmentation/random_translation_1/translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ¥
Pimg_augmentation/random_translation_1/translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ¥
Pimg_augmentation/random_translation_1/translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ¬
Himg_augmentation/random_translation_1/translation_matrix/strided_slice_1StridedSlice5img_augmentation/random_translation_1/concat:output:0Wimg_augmentation/random_translation_1/translation_matrix/strided_slice_1/stack:output:0Yimg_augmentation/random_translation_1/translation_matrix/strided_slice_1/stack_1:output:0Yimg_augmentation/random_translation_1/translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskÈ
<img_augmentation/random_translation_1/translation_matrix/NegNegQimg_augmentation/random_translation_1/translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Iimg_augmentation/random_translation_1/translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¢
Gimg_augmentation/random_translation_1/translation_matrix/zeros_1/packedPackOimg_augmentation/random_translation_1/translation_matrix/strided_slice:output:0Rimg_augmentation/random_translation_1/translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:
Fimg_augmentation/random_translation_1/translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
@img_augmentation/random_translation_1/translation_matrix/zeros_1FillPimg_augmentation/random_translation_1/translation_matrix/zeros_1/packed:output:0Oimg_augmentation/random_translation_1/translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Himg_augmentation/random_translation_1/translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
Fimg_augmentation/random_translation_1/translation_matrix/ones_1/packedPackOimg_augmentation/random_translation_1/translation_matrix/strided_slice:output:0Qimg_augmentation/random_translation_1/translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:
Eimg_augmentation/random_translation_1/translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
?img_augmentation/random_translation_1/translation_matrix/ones_1FillOimg_augmentation/random_translation_1/translation_matrix/ones_1/packed:output:0Nimg_augmentation/random_translation_1/translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
Nimg_augmentation/random_translation_1/translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           ¥
Pimg_augmentation/random_translation_1/translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           ¥
Pimg_augmentation/random_translation_1/translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ¬
Himg_augmentation/random_translation_1/translation_matrix/strided_slice_2StridedSlice5img_augmentation/random_translation_1/concat:output:0Wimg_augmentation/random_translation_1/translation_matrix/strided_slice_2/stack:output:0Yimg_augmentation/random_translation_1/translation_matrix/strided_slice_2/stack_1:output:0Yimg_augmentation/random_translation_1/translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskÊ
>img_augmentation/random_translation_1/translation_matrix/Neg_1NegQimg_augmentation/random_translation_1/translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Iimg_augmentation/random_translation_1/translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¢
Gimg_augmentation/random_translation_1/translation_matrix/zeros_2/packedPackOimg_augmentation/random_translation_1/translation_matrix/strided_slice:output:0Rimg_augmentation/random_translation_1/translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:
Fimg_augmentation/random_translation_1/translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
@img_augmentation/random_translation_1/translation_matrix/zeros_2FillPimg_augmentation/random_translation_1/translation_matrix/zeros_2/packed:output:0Oimg_augmentation/random_translation_1/translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dimg_augmentation/random_translation_1/translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ì
?img_augmentation/random_translation_1/translation_matrix/concatConcatV2Fimg_augmentation/random_translation_1/translation_matrix/ones:output:0Gimg_augmentation/random_translation_1/translation_matrix/zeros:output:0@img_augmentation/random_translation_1/translation_matrix/Neg:y:0Iimg_augmentation/random_translation_1/translation_matrix/zeros_1:output:0Himg_augmentation/random_translation_1/translation_matrix/ones_1:output:0Bimg_augmentation/random_translation_1/translation_matrix/Neg_1:y:0Iimg_augmentation/random_translation_1/translation_matrix/zeros_2:output:0Mimg_augmentation/random_translation_1/translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
5img_augmentation/random_translation_1/transform/ShapeShape\img_augmentation/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:
Cimg_augmentation/random_translation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
Eimg_augmentation/random_translation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Eimg_augmentation/random_translation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
=img_augmentation/random_translation_1/transform/strided_sliceStridedSlice>img_augmentation/random_translation_1/transform/Shape:output:0Limg_augmentation/random_translation_1/transform/strided_slice/stack:output:0Nimg_augmentation/random_translation_1/transform/strided_slice/stack_1:output:0Nimg_augmentation/random_translation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
:img_augmentation/random_translation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
Jimg_augmentation/random_translation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3\img_augmentation/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0Himg_augmentation/random_translation_1/translation_matrix/concat:output:0Fimg_augmentation/random_translation_1/transform/strided_slice:output:0Cimg_augmentation/random_translation_1/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
>img_augmentation/random_flip_1/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
>img_augmentation/random_flip_1/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ø
=img_augmentation/random_flip_1/stateful_uniform_full_int/ProdProdGimg_augmentation/random_flip_1/stateful_uniform_full_int/shape:output:0Gimg_augmentation/random_flip_1/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 
?img_augmentation/random_flip_1/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¿
?img_augmentation/random_flip_1/stateful_uniform_full_int/Cast_1CastFimg_augmentation/random_flip_1/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ö
Gimg_augmentation/random_flip_1/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipPimg_augmentation_random_flip_1_stateful_uniform_full_int_rngreadandskip_resourceHimg_augmentation/random_flip_1/stateful_uniform_full_int/Cast/x:output:0Cimg_augmentation/random_flip_1/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Limg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
Fimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_sliceStridedSliceOimg_augmentation/random_flip_1/stateful_uniform_full_int/RngReadAndSkip:value:0Uimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice/stack:output:0Wimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice/stack_1:output:0Wimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÍ
@img_augmentation/random_flip_1/stateful_uniform_full_int/BitcastBitcastOimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Nimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Pimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Pimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
Himg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1StridedSliceOimg_augmentation/random_flip_1/stateful_uniform_full_int/RngReadAndSkip:value:0Wimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1/stack:output:0Yimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Yimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ñ
Bimg_augmentation/random_flip_1/stateful_uniform_full_int/Bitcast_1BitcastQimg_augmentation/random_flip_1/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0~
<img_augmentation/random_flip_1/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :¬
8img_augmentation/random_flip_1/stateful_uniform_full_intStatelessRandomUniformFullIntV2Gimg_augmentation/random_flip_1/stateful_uniform_full_int/shape:output:0Kimg_augmentation/random_flip_1/stateful_uniform_full_int/Bitcast_1:output:0Iimg_augmentation/random_flip_1/stateful_uniform_full_int/Bitcast:output:0Eimg_augmentation/random_flip_1/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	s
)img_augmentation/random_flip_1/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R Õ
$img_augmentation/random_flip_1/stackPackAimg_augmentation/random_flip_1/stateful_uniform_full_int:output:02img_augmentation/random_flip_1/zeros_like:output:0*
N*
T0	*
_output_shapes

:
2img_augmentation/random_flip_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
4img_augmentation/random_flip_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
4img_augmentation/random_flip_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
,img_augmentation/random_flip_1/strided_sliceStridedSlice-img_augmentation/random_flip_1/stack:output:0;img_augmentation/random_flip_1/strided_slice/stack:output:0=img_augmentation/random_flip_1/strided_slice/stack_1:output:0=img_augmentation/random_flip_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskÚ
Rimg_augmentation/random_flip_1/stateless_random_flip_left_right/control_dependencyIdentity_img_augmentation/random_translation_1/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*]
_classS
QOloc:@img_augmentation/random_translation_1/transform/ImageProjectiveTransformV3*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
Eimg_augmentation/random_flip_1/stateless_random_flip_left_right/ShapeShape[img_augmentation/random_flip_1/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Simg_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Uimg_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Uimg_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Mimg_augmentation/random_flip_1/stateless_random_flip_left_right/strided_sliceStridedSliceNimg_augmentation/random_flip_1/stateless_random_flip_left_right/Shape:output:0\img_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice/stack:output:0^img_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice/stack_1:output:0^img_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskì
^img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/shapePackVimg_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:¡
\img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ¡
\img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?î
uimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter5img_augmentation/random_flip_1/strided_slice:output:0* 
_output_shapes
::·
uimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
qimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2gimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0{img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0~img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
\img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/subSubeimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/max:output:0eimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ï
\img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/mulMulzimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0`img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
Ximg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniformAddV2`img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0eimg_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Oimg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Oimg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Oimg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :é
Mimg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shapePackVimg_augmentation/random_flip_1/stateless_random_flip_left_right/strided_slice:output:0Ximg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shape/1:output:0Ximg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shape/2:output:0Ximg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Â
Gimg_augmentation/random_flip_1/stateless_random_flip_left_right/ReshapeReshape\img_augmentation/random_flip_1/stateless_random_flip_left_right/stateless_random_uniform:z:0Vimg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
Eimg_augmentation/random_flip_1/stateless_random_flip_left_right/RoundRoundPimg_augmentation/random_flip_1/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nimg_augmentation/random_flip_1/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:È
Iimg_augmentation/random_flip_1/stateless_random_flip_left_right/ReverseV2	ReverseV2[img_augmentation/random_flip_1/stateless_random_flip_left_right/control_dependency:output:0Wimg_augmentation/random_flip_1/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Cimg_augmentation/random_flip_1/stateless_random_flip_left_right/mulMulIimg_augmentation/random_flip_1/stateless_random_flip_left_right/Round:y:0Rimg_augmentation/random_flip_1/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Eimg_augmentation/random_flip_1/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Cimg_augmentation/random_flip_1/stateless_random_flip_left_right/subSubNimg_augmentation/random_flip_1/stateless_random_flip_left_right/sub/x:output:0Iimg_augmentation/random_flip_1/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
Eimg_augmentation/random_flip_1/stateless_random_flip_left_right/mul_1MulGimg_augmentation/random_flip_1/stateless_random_flip_left_right/sub:z:0[img_augmentation/random_flip_1/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Cimg_augmentation/random_flip_1/stateless_random_flip_left_right/addAddV2Gimg_augmentation/random_flip_1/stateless_random_flip_left_right/mul:z:0Iimg_augmentation/random_flip_1/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bimg_augmentation/random_contrast_1/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Bimg_augmentation/random_contrast_1/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Aimg_augmentation/random_contrast_1/stateful_uniform_full_int/ProdProdKimg_augmentation/random_contrast_1/stateful_uniform_full_int/shape:output:0Kimg_augmentation/random_contrast_1/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 
Cimg_augmentation/random_contrast_1/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Ç
Cimg_augmentation/random_contrast_1/stateful_uniform_full_int/Cast_1CastJimg_augmentation/random_contrast_1/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: æ
Kimg_augmentation/random_contrast_1/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipTimg_augmentation_random_contrast_1_stateful_uniform_full_int_rngreadandskip_resourceLimg_augmentation/random_contrast_1/stateful_uniform_full_int/Cast/x:output:0Gimg_augmentation/random_contrast_1/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Pimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Rimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Rimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Jimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_sliceStridedSliceSimg_augmentation/random_contrast_1/stateful_uniform_full_int/RngReadAndSkip:value:0Yimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice/stack:output:0[img_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice/stack_1:output:0[img_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÕ
Dimg_augmentation/random_contrast_1/stateful_uniform_full_int/BitcastBitcastSimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Rimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Timg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Timg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
Limg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1StridedSliceSimg_augmentation/random_contrast_1/stateful_uniform_full_int/RngReadAndSkip:value:0[img_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1/stack:output:0]img_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_1:output:0]img_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ù
Fimg_augmentation/random_contrast_1/stateful_uniform_full_int/Bitcast_1BitcastUimg_augmentation/random_contrast_1/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
@img_augmentation/random_contrast_1/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :À
<img_augmentation/random_contrast_1/stateful_uniform_full_intStatelessRandomUniformFullIntV2Kimg_augmentation/random_contrast_1/stateful_uniform_full_int/shape:output:0Oimg_augmentation/random_contrast_1/stateful_uniform_full_int/Bitcast_1:output:0Mimg_augmentation/random_contrast_1/stateful_uniform_full_int/Bitcast:output:0Iimg_augmentation/random_contrast_1/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	w
-img_augmentation/random_contrast_1/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R á
(img_augmentation/random_contrast_1/stackPackEimg_augmentation/random_contrast_1/stateful_uniform_full_int:output:06img_augmentation/random_contrast_1/zeros_like:output:0*
N*
T0	*
_output_shapes

:
6img_augmentation/random_contrast_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
8img_augmentation/random_contrast_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
8img_augmentation/random_contrast_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¦
0img_augmentation/random_contrast_1/strided_sliceStridedSlice1img_augmentation/random_contrast_1/stack:output:0?img_augmentation/random_contrast_1/strided_slice/stack:output:0Aimg_augmentation/random_contrast_1/strided_slice/stack_1:output:0Aimg_augmentation/random_contrast_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
Aimg_augmentation/random_contrast_1/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
?img_augmentation/random_contrast_1/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
?img_augmentation/random_contrast_1/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?Õ
Ximg_augmentation/random_contrast_1/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter9img_augmentation/random_contrast_1/strided_slice:output:0* 
_output_shapes
::
Ximg_augmentation/random_contrast_1/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :û
Timg_augmentation/random_contrast_1/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Jimg_augmentation/random_contrast_1/stateless_random_uniform/shape:output:0^img_augmentation/random_contrast_1/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0bimg_augmentation/random_contrast_1/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0aimg_augmentation/random_contrast_1/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: û
?img_augmentation/random_contrast_1/stateless_random_uniform/subSubHimg_augmentation/random_contrast_1/stateless_random_uniform/max:output:0Himg_augmentation/random_contrast_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
?img_augmentation/random_contrast_1/stateless_random_uniform/mulMul]img_augmentation/random_contrast_1/stateless_random_uniform/StatelessRandomUniformV2:output:0Cimg_augmentation/random_contrast_1/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ô
;img_augmentation/random_contrast_1/stateless_random_uniformAddV2Cimg_augmentation/random_contrast_1/stateless_random_uniform/mul:z:0Himg_augmentation/random_contrast_1/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
2img_augmentation/random_contrast_1/adjust_contrastAdjustContrastv2Gimg_augmentation/random_flip_1/stateless_random_flip_left_right/add:z:0?img_augmentation/random_contrast_1/stateless_random_uniform:z:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
;img_augmentation/random_contrast_1/adjust_contrast/IdentityIdentity;img_augmentation/random_contrast_1/adjust_contrast:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype0æ
conv2d/Conv2DConv2DDimg_augmentation/random_contrast_1/adjust_contrast/Identity:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? *
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? 
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0¹
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ?? : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ?? §
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Å
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ		@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@«
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Æ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0´
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Å
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0z
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
avg_pool/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
avg_pool/MaxMaxre_lu_2/Relu:activations:0'avg_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
top_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?
top_dropout/dropout/MulMulavg_pool/Max:output:0"top_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
top_dropout/dropout/ShapeShapeavg_pool/Max:output:0*
T0*
_output_shapes
:¤
0top_dropout/dropout/random_uniform/RandomUniformRandomUniform"top_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0g
"top_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ê
 top_dropout/dropout/GreaterEqualGreaterEqual9top_dropout/dropout/random_uniform/RandomUniform:output:0+top_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
top_dropout/dropout/CastCast$top_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
top_dropout/dropout/Mul_1Multop_dropout/dropout/Mul:z:0top_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
pred/MatMul/ReadVariableOpReadVariableOp#pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
pred/MatMulMatMultop_dropout/dropout/Mul_1:z:0"pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
pred/BiasAdd/ReadVariableOpReadVariableOp$pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
pred/BiasAddBiasAddpred/MatMul:product:0#pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
pred/SoftmaxSoftmaxpred/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-pred/kernel/Regularizer/Square/ReadVariableOpReadVariableOp#pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
pred/kernel/Regularizer/SquareSquare5pred/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@n
pred/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
pred/kernel/Regularizer/SumSum"pred/kernel/Regularizer/Square:y:0&pred/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: b
pred/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
pred/kernel/Regularizer/mulMul&pred/kernel/Regularizer/mul/x:output:0$pred/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitypred/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOpL^img_augmentation/random_contrast_1/stateful_uniform_full_int/RngReadAndSkipH^img_augmentation/random_flip_1/stateful_uniform_full_int/RngReadAndSkipC^img_augmentation/random_rotation_1/stateful_uniform/RngReadAndSkipF^img_augmentation/random_translation_1/stateful_uniform/RngReadAndSkipH^img_augmentation/random_translation_1/stateful_uniform_1/RngReadAndSkip^pred/BiasAdd/ReadVariableOp^pred/MatMul/ReadVariableOp.^pred/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2
Kimg_augmentation/random_contrast_1/stateful_uniform_full_int/RngReadAndSkipKimg_augmentation/random_contrast_1/stateful_uniform_full_int/RngReadAndSkip2
Gimg_augmentation/random_flip_1/stateful_uniform_full_int/RngReadAndSkipGimg_augmentation/random_flip_1/stateful_uniform_full_int/RngReadAndSkip2
Bimg_augmentation/random_rotation_1/stateful_uniform/RngReadAndSkipBimg_augmentation/random_rotation_1/stateful_uniform/RngReadAndSkip2
Eimg_augmentation/random_translation_1/stateful_uniform/RngReadAndSkipEimg_augmentation/random_translation_1/stateful_uniform/RngReadAndSkip2
Gimg_augmentation/random_translation_1/stateful_uniform_1/RngReadAndSkipGimg_augmentation/random_translation_1/stateful_uniform_1/RngReadAndSkip2:
pred/BiasAdd/ReadVariableOppred/BiasAdd/ReadVariableOp28
pred/MatMul/ReadVariableOppred/MatMul/ReadVariableOp2^
-pred/kernel/Regularizer/Square/ReadVariableOp-pred/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_2_layer_call_fn_40605

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38710
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

h
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38025

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
I
-__inference_max_pooling2d_layer_call_fn_40454

inputs
identityÙ
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
GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_38578
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
å

,__inference_try_drop_fur_layer_call_fn_39421

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
identity¢StatefulPartitionedCallÒ
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
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_38878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
©

ú
A__inference_conv2d_layer_call_and_return_conditional_losses_38752

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
¸

#__inference_signature_wrapper_40015
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
identity¢StatefulPartitionedCall¬
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
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_38014o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
æ
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_38805

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
¼
C
'__inference_re_lu_2_layer_call_fn_40646

inputs
identity¸
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
GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_38838h
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
Þ
P
4__inference_random_translation_1_layer_call_fn_40871

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_random_translation_1_layer_call_and_return_conditional_losses_38031j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
\
@__inference_re_lu_layer_call_and_return_conditional_losses_40449

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

h
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40748

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_2_layer_call_fn_40592

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_38679
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

Å
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_38429

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq¿Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¡
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Î
3__inference_batch_normalization_layer_call_fn_40403

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_38558
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
ëz

__inference__traced_save_41355
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop*
&savev2_pred_kernel_read_readvariableop(
$savev2_pred_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_random_rotation_1_statevar_read_readvariableop	<
8savev2_random_translation_1_statevar_read_readvariableop	5
1savev2_random_flip_1_statevar_read_readvariableop	9
5savev2_random_contrast_1_statevar_read_readvariableop	3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1
-savev2_adam_pred_kernel_m_read_readvariableop/
+savev2_adam_pred_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1
-savev2_adam_pred_kernel_v_read_readvariableop/
+savev2_adam_pred_bias_v_read_readvariableop
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
: "
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*Ç!
value½!Bº!>B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-3/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHì
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop&savev2_pred_kernel_read_readvariableop$savev2_pred_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_random_rotation_1_statevar_read_readvariableop8savev2_random_translation_1_statevar_read_readvariableop1savev2_random_flip_1_statevar_read_readvariableop5savev2_random_contrast_1_statevar_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop-savev2_adam_pred_kernel_m_read_readvariableop+savev2_adam_pred_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop-savev2_adam_pred_kernel_v_read_readvariableop+savev2_adam_pred_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *L
dtypesB
@2>					
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

identity_1Identity_1:output:0*Û
_input_shapesÉ
Æ: :		 : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@:: : : : : : : : : :::::		 : : : : @:@:@:@:@@:@:@:@:@::		 : : : : @:@:@:@:@@:@:@:@:@:: 2(
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:		 : #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:,*(
&
_output_shapes
:@@: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@:$. 

_output_shapes

:@: /

_output_shapes
::,0(
&
_output_shapes
:		 : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
: @: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:,8(
&
_output_shapes
:@@: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:$< 

_output_shapes

:@: =

_output_shapes
::>

_output_shapes
: 
è

,__inference_try_drop_fur_layer_call_fn_38921
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
identity¢StatefulPartitionedCallÓ
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
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_38878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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

ñ
,__inference_try_drop_fur_layer_call_fn_39226
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	#
	unknown_3:		 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39122o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Á
Ë
0__inference_img_augmentation_layer_call_fn_40033

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38457y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¿
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40641

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
¹
K
/__inference_max_pooling2d_1_layer_call_fn_40555

inputs
identityÛ
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
GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_38654
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

ð
,__inference_try_drop_fur_layer_call_fn_39474

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	#
	unknown_3:		 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39122o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ8
pred0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ä
ï
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
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
Þ
layer-0
layer-1
layer-2
layer-3
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_sequential
»

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
»

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
§
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ô
	iter
beta_1
beta_2

decay
learning_rate%m«&m¬.m­/m®Dm¯Em°Mm±Nm²cm³dm´lmµmm¶	m·	m¸%v¹&vº.v»/v¼Dv½Ev¾Mv¿NvÀcvÁdvÂlvÃmvÄ	vÅ	vÆ"
	optimizer
¸
%0
&1
.2
/3
04
15
D6
E7
M8
N9
O10
P11
c12
d13
l14
m15
n16
o17
18
19"
trackable_list_wrapper

%0
&1
.2
/3
D4
E5
M6
N7
c8
d9
l10
m11
12
13"
trackable_list_wrapper
(
0"
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_try_drop_fur_layer_call_fn_38921
,__inference_try_drop_fur_layer_call_fn_39421
,__inference_try_drop_fur_layer_call_fn_39474
,__inference_try_drop_fur_layer_call_fn_39226À
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
ê2ç
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39559
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39968
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39291
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39364À
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
ËBÈ
 __inference__wrapped_model_38014input_1"
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
serving_default"
signature_map
 "
trackable_list_wrapper
Ã
	variables
trainable_variables
regularization_losses
 	keras_api
¡_random_generator
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨_random_generator
©__call__
+ª&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯_random_generator
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶_random_generator
·__call__
+¸&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
2
0__inference_img_augmentation_layer_call_fn_38049
0__inference_img_augmentation_layer_call_fn_40020
0__inference_img_augmentation_layer_call_fn_40033
0__inference_img_augmentation_layer_call_fn_38481À
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
ú2÷
K__inference_img_augmentation_layer_call_and_return_conditional_losses_40037
K__inference_img_augmentation_layer_call_and_return_conditional_losses_40358
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38489
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38505À
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
':%		 2conv2d/kernel
: 2conv2d/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_conv2d_layer_call_fn_40367¢
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
ë2è
A__inference_conv2d_layer_call_and_return_conditional_losses_40377¢
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
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
<
.0
/1
02
13"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
¤2¡
3__inference_batch_normalization_layer_call_fn_40390
3__inference_batch_normalization_layer_call_fn_40403´
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
Ú2×
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40421
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40439´
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
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_re_lu_layer_call_fn_40444¢
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
@__inference_re_lu_layer_call_and_return_conditional_losses_40449¢
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
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_max_pooling2d_layer_call_fn_40454¢
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
ò2ï
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_40459¢
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
):' @2conv2d_1/kernel
:@2conv2d_1/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_conv2d_1_layer_call_fn_40468¢
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40478¢
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
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
<
M0
N1
O2
P3"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
¨2¥
5__inference_batch_normalization_1_layer_call_fn_40491
5__inference_batch_normalization_1_layer_call_fn_40504´
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
Þ2Û
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_40522
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_40540´
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
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_re_lu_1_layer_call_fn_40545¢
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
ì2é
B__inference_re_lu_1_layer_call_and_return_conditional_losses_40550¢
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
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_max_pooling2d_1_layer_call_fn_40555¢
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
ô2ñ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40560¢
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
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_conv2d_2_layer_call_fn_40569¢
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40579¢
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
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
<
l0
m1
n2
o3"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
¨2¥
5__inference_batch_normalization_2_layer_call_fn_40592
5__inference_batch_normalization_2_layer_call_fn_40605´
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
Þ2Û
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40623
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40641´
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
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_re_lu_2_layer_call_fn_40646¢
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
ì2é
B__inference_re_lu_2_layer_call_and_return_conditional_losses_40651¢
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
µ
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_avg_pool_layer_call_fn_40656¢
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
C__inference_avg_pool_layer_call_and_return_conditional_losses_40662¢
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
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_top_dropout_layer_call_fn_40667
+__inference_top_dropout_layer_call_fn_40672´
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
Ê2Ç
F__inference_top_dropout_layer_call_and_return_conditional_losses_40677
F__inference_top_dropout_layer_call_and_return_conditional_losses_40689´
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
:@2pred/kernel
:2	pred/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_pred_layer_call_fn_40704¢
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
é2æ
?__inference_pred_layer_call_and_return_conditional_losses_40721¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
²2¯
__inference_loss_fn_0_40732
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
00
11
O2
P3
n4
o5"
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
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
#__inference_signature_wrapper_40015input_1"
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
/

_generator"
_generic_user_object
 2
1__inference_random_rotation_1_layer_call_fn_40737
1__inference_random_rotation_1_layer_call_fn_40744´
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
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40748
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40866´
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
p

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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
/

_generator"
_generic_user_object
¦2£
4__inference_random_translation_1_layer_call_fn_40871
4__inference_random_translation_1_layer_call_fn_40878´
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
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
O__inference_random_translation_1_layer_call_and_return_conditional_losses_40882
O__inference_random_translation_1_layer_call_and_return_conditional_losses_40998´
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
p

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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
«	variables
¬trainable_variables
­regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
/

_generator"
_generic_user_object
2
-__inference_random_flip_1_layer_call_fn_41003
-__inference_random_flip_1_layer_call_fn_41010´
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
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_random_flip_1_layer_call_and_return_conditional_losses_41014
H__inference_random_flip_1_layer_call_and_return_conditional_losses_41072´
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
p

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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
/

_generator"
_generic_user_object
 2
1__inference_random_contrast_1_layer_call_fn_41077
1__inference_random_contrast_1_layer_call_fn_41084´
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
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_41088
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_41129´
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
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
<
0
1
2
3"
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
00
11"
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
O0
P1"
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
n0
o1"
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
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
 	variables
¡	keras_api"
_tf_keras_metric
c

¢total

£count
¤
_fn_kwargs
¥	variables
¦	keras_api"
_tf_keras_metric
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
/
§
_state_var"
_generic_user_object
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
/
¨
_state_var"
_generic_user_object
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
/
©
_state_var"
_generic_user_object
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
/
ª
_state_var"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
 	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¢0
£1"
trackable_list_wrapper
.
¥	variables"
_generic_user_object
&:$	2random_rotation_1/StateVar
):'	2random_translation_1/StateVar
": 	2random_flip_1/StateVar
&:$	2random_contrast_1/StateVar
,:*		 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
,:* 2 Adam/batch_normalization/gamma/m
+:) 2Adam/batch_normalization/beta/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
": @2Adam/pred/kernel/m
:2Adam/pred/bias/m
,:*		 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
,:* 2 Adam/batch_normalization/gamma/v
+:) 2Adam/batch_normalization/beta/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
": @2Adam/pred/kernel/v
:2Adam/pred/bias/v¦
 __inference__wrapped_model_38014%&./01DEMNOPcdlmno:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ
ª "+ª(
&
pred
predÿÿÿÿÿÿÿÿÿÌ
C__inference_avg_pool_layer_call_and_return_conditional_losses_40662R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 £
(__inference_avg_pool_layer_call_fn_40656wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_40522MNOPM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_40540MNOPM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ã
5__inference_batch_normalization_1_layer_call_fn_40491MNOPM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_1_layer_call_fn_40504MNOPM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ë
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40623lmnoM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_40641lmnoM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ã
5__inference_batch_normalization_2_layer_call_fn_40592lmnoM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_2_layer_call_fn_40605lmnoM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40421./01M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40439./01M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Á
3__inference_batch_normalization_layer_call_fn_40390./01M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Á
3__inference_batch_normalization_layer_call_fn_40403./01M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ³
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40478lDE7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ		@
 
(__inference_conv2d_1_layer_call_fn_40468_DE7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ		@³
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40579lcd7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_conv2d_2_layer_call_fn_40569_cd7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@³
A__inference_conv2d_layer_call_and_return_conditional_losses_40377n%&9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ?? 
 
&__inference_conv2d_layer_call_fn_40367a%&9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ?? Õ
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38489R¢O
H¢E
;8
random_rotation_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ß
K__inference_img_augmentation_layer_call_and_return_conditional_losses_38505§¨©ªR¢O
H¢E
;8
random_rotation_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ã
K__inference_img_augmentation_layer_call_and_return_conditional_losses_40037tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Í
K__inference_img_augmentation_layer_call_and_return_conditional_losses_40358~§¨©ªA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¬
0__inference_img_augmentation_layer_call_fn_38049xR¢O
H¢E
;8
random_rotation_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ·
0__inference_img_augmentation_layer_call_fn_38481§¨©ªR¢O
H¢E
;8
random_rotation_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ
0__inference_img_augmentation_layer_call_fn_40020gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¥
0__inference_img_augmentation_layer_call_fn_40033q§¨©ªA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_40732¢

¢ 
ª " í
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40560R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_1_layer_call_fn_40555R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_40459R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_max_pooling2d_layer_call_fn_40454R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
?__inference_pred_layer_call_and_return_conditional_losses_40721^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
$__inference_pred_layer_call_fn_40704Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀ
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_41088p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ä
L__inference_random_contrast_1_layer_call_and_return_conditional_losses_41129tª=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
1__inference_random_contrast_1_layer_call_fn_41077c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
1__inference_random_contrast_1_layer_call_fn_41084gª=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ¼
H__inference_random_flip_1_layer_call_and_return_conditional_losses_41014p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 À
H__inference_random_flip_1_layer_call_and_return_conditional_losses_41072t©=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_random_flip_1_layer_call_fn_41003c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
-__inference_random_flip_1_layer_call_fn_41010g©=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿÀ
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40748p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ä
L__inference_random_rotation_1_layer_call_and_return_conditional_losses_40866t§=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
1__inference_random_rotation_1_layer_call_fn_40737c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
1__inference_random_rotation_1_layer_call_fn_40744g§=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿÃ
O__inference_random_translation_1_layer_call_and_return_conditional_losses_40882p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ç
O__inference_random_translation_1_layer_call_and_return_conditional_losses_40998t¨=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
4__inference_random_translation_1_layer_call_fn_40871c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
4__inference_random_translation_1_layer_call_fn_40878g¨=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ®
B__inference_re_lu_1_layer_call_and_return_conditional_losses_40550h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ		@
 
'__inference_re_lu_1_layer_call_fn_40545[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª " ÿÿÿÿÿÿÿÿÿ		@®
B__inference_re_lu_2_layer_call_and_return_conditional_losses_40651h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_re_lu_2_layer_call_fn_40646[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¬
@__inference_re_lu_layer_call_and_return_conditional_losses_40449h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ?? 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ?? 
 
%__inference_re_lu_layer_call_fn_40444[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ?? 
ª " ÿÿÿÿÿÿÿÿÿ?? ´
#__inference_signature_wrapper_40015%&./01DEMNOPcdlmnoE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿ"+ª(
&
pred
predÿÿÿÿÿÿÿÿÿ¦
F__inference_top_dropout_layer_call_and_return_conditional_losses_40677\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_top_dropout_layer_call_and_return_conditional_losses_40689\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_top_dropout_layer_call_fn_40667O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_top_dropout_layer_call_fn_40672O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@Ï
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39291%&./01DEMNOPcdlmnoB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ×
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39364§¨©ª%&./01DEMNOPcdlmnoB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Î
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39559%&./01DEMNOPcdlmnoA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ö
G__inference_try_drop_fur_layer_call_and_return_conditional_losses_39968§¨©ª%&./01DEMNOPcdlmnoA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
,__inference_try_drop_fur_layer_call_fn_38921v%&./01DEMNOPcdlmnoB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ®
,__inference_try_drop_fur_layer_call_fn_39226~§¨©ª%&./01DEMNOPcdlmnoB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¥
,__inference_try_drop_fur_layer_call_fn_39421u%&./01DEMNOPcdlmnoA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ­
,__inference_try_drop_fur_layer_call_fn_39474}§¨©ª%&./01DEMNOPcdlmnoA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ