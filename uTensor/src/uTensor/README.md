# uTensor Runtime

The uTensor runtime is made up of two major components, 1) the uTensor Core which contains the basic data structures, interfaces, and Types required to meet the uTensor performance runtime contract, and 2) the uTensor Library which serves as a series of reasonable default implementations built on top of the uTensor Core. The build system compiles these two components separately, allowing users to easily extend and override implementations, such as custom memory managers, tensors, operators, and error handlers, built on top of the uTensor core.

For the rest of this document we will use the lowercase form of *tensor* to describe any tensor-like object that implements the `TensorInterface`, and the uppercase form of `Tensor` to describe a allocated `Handle` (basically a pointer like reference) bound to a particular tensor object managed by uTensor.  

## uTensor Core

The uTensor Core is exactly that, it is the heart, definition, and ruthless contractual obligations that lets the runtime guarantee things like memory safety and model updateability, as well as a consistent user experience. Despite appearing like a high level language, the uTensor core compiles to an extremely small footprint somewhere between 1 kB and 2 KB (plus ~1KB for the rest uTensor library).

For the rest of the discussion on the uTensor core, we will explain what each part does under the hood, and what is expected of the user if they want to extend any implementations.

### Events and Errors and their Handlers

In general, runtime type information (RTTI) is expensive for tiny systems. Rather than forcing users to compile their code with RTTI enabled and bear the cost, we found a neat way to give uTensor the ability to identify events dynamically at runtime with configurable cost! Basically using only C++11 language features, we just hash the signature of an event type at compile time and store this as a, mostly, unique ID inside the `Event` objects. A nice byproduct is these IDs remain the same across builds and machines: unless someone explicitly changes the signature of an `Event`, the user doesnt need to manually specify some magic number associated with each event. The size of this ID is configurable, and can be 1 byte, 2 bytes, or 4 bytes depending on how many unique event types you need, even though the 4 byte version is probably way overkill for small devices.

This IDs are pretty useful when debugging remote deployments. All you have to do is `grep` your source code for `DECLARE_EVENT({EVENT_NAME})` or `DECLARE_ERROR({ERROR_NAME})`, run the same hash function on the `{EVENT_NAME}`s, and store this in a simple `map(hash({EVENT_NAME}) => {EVENT_NAME})`. Then if an `Event` or `Error` occurs you just have to query this map.

Alternatively, this Python snippet will scrape codebases for all basically defined `Event`s and `Error`s and store them in a Python dictionary:

```python
import numpy as np
import glob
import re

from collections import defaultdict
from pprint import pprint


def mHash_fnv1a(mStr):
  np.seterr(over='ignore')
  val_32_const = np.uint32(0x811c9dc5)
  prime_32_const = np.uint32(0x1000193)
  value = val_32_const
  for c in mStr:
      value = (value ^ np.uint32(ord(c))) * prime_32_const
  return value

def get_target_files():
  x = glob.glob('**/*.[ch]pp', recursive=True)
  return x

def get_event_map():
  tgts = get_target_files()
  event_names = []
  event_map = defaultdict(list)
  for f in tgts:
    with open(f) as fp:
      for line in fp:
        m = re.match("\s*DECLARE_\w+\((\w+)\)", line)
        if m:
          #print(m)
          event_names.append(m.group(1))

  for evt in event_names:
    x = mHash_fnv1a(evt)
    event_map[x].append(evt)
  pprint(event_map)
  return event_map

if __name__ == "__main__":
  x = get_event_map()
```

### Basic Types
#### IntegralType

The `IntegralType` is basically an intermediate type that can take on `{u}int8_t`, `{u}int16_t`, `{u}int32_t`, or `float32` depending on the corresponding data type being written to/read from it. It can also behave as a reference to the target types, as is the case for the write interface for the `TensorInterface`. Really this class is only used by the reference operators and reference TensorInterface R/W interface, but from our initial experiments it actually compiles to relatively efficient machine code for Arm targets. 

As a user, you will **almost NEVER** have a reason to use this type directly, instead opt for `static_cast`s at functions/functors boundaries that deal with these types. It makes the code way more readable and better captures intent. 

#### uTensor Strings

We get it, C-strings are extremely useful for user interfaces and early debugging of code. However, when dealing with KB sized binaries even a small number of short strings add up to a non-trivial number of bytes. The `uTensor::string` class is basically a proxy string that can do quick string comparisons for various data structure by first hashing the string. Eventually, we plan on doing the hash at compile time so we can quickly remove all references to string data between debug and release builds with a simple build flag.

It is better to thing about `uTensor::string` as an identifier rather than a string.

#### TensorShape

`TensorShape` is exactly that, it describes the shape of a tensor as well as some basic helper functions like how many elements represented by this shape. `TensorShape` is **a fixed size object**, at the moment it always has exactly 4 stored dimensions even when some are not used, and furthermore it does not have any virtual functions.

`TensorShape` also as the ability to transform 2D, 3D, and 4D, indices into a linear index based on it's shape. This is super helpful when writing `FastOperators` as it is simple low-cost syntactic sugar that keeps pointer arithmetic clean.

#### Quantization Primitives

The Quantization primitives represent per-channel and per-tensor symmetric quantization schemes shared with https://www.tensorflow.org/lite/performance/quantization_spec. This generally involves allocating some small buffer in the RAM allocator so we can compute and maintain scaling factors and arithmetic shifts. Choosing when to allocate, compute, and deallocate can directly impact performance. For example, allocating once and computing these values on model setup and deallocating on tear down trades RAM for improved inference times. Whereas allocating, computing, and deallocating on each inference trades inference speed for lower peak RAM footprints.

### Memory Allocator Interfaces

Probably the **two most important concepts in uTensor** are those of the Memory Allocators and `Handles`

From the name classes that extend the `AllocatorInterface` can do really anything a normal memory manager can, such as allocating and deallocating raw data and querying the amount of space left. However, they way this differs from standard implementations is the uTensor notion of `Handles` and how they allow the allocator to do things not normally permitted in systems without virtual memory. 

In uTensor, `Handle`s are basically unique (non-copyable) pointer-like proxy reference **bound** to some allocated region in the memory manager. Under the hood, this is just a void pointer, but the user cannot access it directly. Instead, you must you must dereference the `Handle` object to get access to the underlying pointer to data (automatically a double dereference). This is important because the memory manager keeps track of `Handle`s internally and can move the underlying data blocks around without breaking the user code! Making the `Handle`s non-copyable means the memory manager only needs to record one instance per bound region which saves a good chunk of space and speeds up the query. Likewise in user-land, if `Handles` are passed around by reference then if the Allocator moved the underlying data, all user-land references effectively get *notified*. 

The most important part about handles:
`Handle`s bound to an allocated region in a memory manager are guaranteed to be valid until expressly unbound. In other words the allocator is **not-allowed** to deallocate a bound region, but it is allowed to move it around as long as it updates the associated `Handle`. We eventually plan on adding a `Scheduler` which has the ability to move around bound data in the memory allocator given some memory optimal *plan*, and the `Handle`s let us do that as well as minimize fragmentation without breaking user-space coherency. 

### TensorInterface and the tensor lifecycle

In uTensor, tensor-like objects that implement `TensorInterface` get two things 1) automatic uTensor-internal lifecycle management and 2) a consistent user interface. `new` and `delete` are overwritten such that child classes of `TensorInterface` are automatically placed in a uTensor managed allocator with all associated behavior intact. Furthermore, on construction these objects get automatically registered in the uTensor runtime context, which is useful for profiling, debugging, and building scheduling systems. 

The default user interface provides various operations expected of ALL tensor-like objects, such as shape queries, and reading/writing values. This R/W interface defaults to passing around `IntegralValue`s which can be roughly controlled via static casting of values (described in the `IntegralValue` section). 

Although the default R/W interface is super clear and compiles to somewhat efficient code, it does not allow for the knuckle-bleeding rip-your-face-off drag-race speeds of accessing the underlying blocks of data directly which is often needed in latency optimized models. For that, we provide a protected interface for querying a span of the underlying data directly for reading/writing. This interface is meant for advanced capabilities and can only be accessed by friend tags like `FastOperator`. All the optimized operators use this interface plus pointer indexing, whereas the reference operators use the N-D tensor indexing with default R/W for clarity.


### Tensors, just Handles bound to objects implementing TensorInteface

Up until now we have described tensors as "tensor-like objects" that implement `TensorInterface`, but given our newfound knowledge of the memory allocator system we can extend the concept of `Handle`s to specialize on these tensor-like objects and make them fully *managed* by the uTensor runtime. 
`Tensors` are exactly that, they are `Handles` that exclusively bind to `TensorInterface` objects managed in the allocators, and therefore get all the benefits of being a bound region. 

Furthermore, `Tensors` provide some extra helper functions that make it feel more like we are operating on general tensor-like objects, rather than some externally referenced object in some magical memory manager. For example, constructing/assigning a `Tensor` to a dynamically allocated `TensorInterface*` will automatically bind it in the memory manager! Also, dereferencing a `Tensor` handle will automatically cast the referenced data to `TensorInterface*` so you don't have to worry about polymorphism corner cases. 

Like `Handles`, when a `Tensor` goes out of scope, or more generally when it's destructor gets called, the underlying tensor object is unbound and deallocated in the associated memory allocator.

### TensorMap

`TensorMap`s are nothing more than an ordered map of `uTensor::string`s to `Tensor` references. Tensors can be looked up by "name", or more accurately ID. These are used heavily in the operators to map named inputs to tensors. Seeing `mOperator.setInputs({{mOperator::a, tensor1}, {mOperator::filter, f_tensor_891293}, {mOperator::bias, cowsay_tensor}}))` it's much clearer which tensor is being used for what purpose in the operator without having to jump to the operator class declaration.

### OperatorInterface

The `OperatorInterface`, in all its glory, is nothing more than a fixed-size `TensorMap` for inputs and another for outputs, the setters for these two maps, plus a pure virtual `compute()` method that gets called whenever a user `eval`s an operator. Although, writing a custom operator just involves implementing this `compute()` method in a child class, we find it helpful to make the names of the expected inputs/outputs additionally available as public enums in the child classes. This way we can clearly describe which `Tensor`s get mapped to which operator parameter, and also `ctag` jump around in the models more easily.

```cpp
class MyOperator : public OperatorInterface<num_inputs, num_outputs> {
 public:
  enum names_in : uint8_t { in, filter }; // these are named IDs for the inputs
  enum names_out : uint8_t { out };   // these are named IDs for the outputs

  // Optional constructor

 protected:
  virtual void compute() {
    // Operator interface maintains a 2 maps of tensor names to tensors, one for inputs, and one for outputs 
    my_kernel(outputs[out].tensor(), inputs[in].tensor(), inputs[filter].tensor());
  }
};
...
MyOperator myOp;
myOp
  .setInputs({{myOp::in, tensor1}, {myOp::filter, tensor2}}) //Bind tensors to input names in op
  .setOutputs({{myOp::out, tensor_tardis_392}})              //Bind tensors to output names in op
  .eval();                                                   //Evaluate operator given inputs/outputs
```

## uTensor Lib

The uTensor Library serves as a series of reasonable default implementations built on top of the uTensor Core. These will be described in detail further down, but includes some examples such as:

- errorHandlers
- allocators
- contexts
- ops
  - legacy
  - optimized
  - symmetric_quantization
  - reference
- tensors

### Error Handlers

The `SimpleErrorHandler` is literally just that, it maintains a fixed sized queue for events and allows users to override the default `spin-wait` behavior of `onError` by passing an `ErrorHandler` callback functor. Although simple, this error handler is used heavily in the uTensor tests both in verifying error conditions and in checking partial ordering of various events notified by the uTensor components.

```cpp
SimpleErrorHandler errH(50); // Maintain a history of 50 events
Context::get_default_context()->set_ErrorHandler(&errH);
...
// A bunch of allocations
...

// Check to make sure a rebalance has occurred inside our allocator
bool has_rebalanced = std::find(errH.begin(), errH.end(), localCircularArenaAllocatorRebalancingEvent()) != errH.end();
```

### Allocators

The default allocator is a fixed sized circular arena allocator with statically configurable internal addressability. This means users can scale back the internal cost of representing the allocator if they only need to allocate a small number of small blocks, or scale up to handle much larger allocations which is fairly common with image processing. This allocator will handle object alignment internally, meaning it's possible to place objects and functors in this memory space. 

Allocated blocks that are **bound** to a `Handle` are guaranteed to be valid after rebalance. This data structure will move the underlying blocks around to minimize fragmentation so that the maximum amount of space is available for the next allocation. 

Allocated blocks that are **unbound** (not associated with a handle) are not guaranteed to be valid after a rebalance. It will be up to the user to query the `localCircularArenaAllocator` to check for pointer validity before reading/writing. Not checking will lead to undefined access behavior.

#### Events and Errors

- `OutOfMemBoundsError`: [Error], thrown if user attempts to access non managed region of memory, e.x. Handle is corrupted
- `MetaHeaderNotFound`: [Event], notification that region is not found inside the data structure, used in Testing
- `InvalidBoundRegionState`: [Error], thrown if attempted to bind handle with unallocated region
- `OutOfMemError`: [Error], thrown if memory manager doesnt have enough space to fulfill the allocation request
- `localCircularArenaAllocatorConstructed`: [Event], notification that allocator has been constructed
- `localCircularArenaAllocatorRebalancing`: [Event], notification that allocator rebalancing has occured

### Contexts

The default context is meant to be a bare minimum implemenation, as all it does is maintain a reference to the two required `allocatorInterfaces`, the metadata allocator and the RAM allocator, and an `errorHandler`. Extensions on this could be maintaining the local graph structure which allows uTensor to run as a virtual machine instead of a static code model.

Tensors, operators, and models use this context to maintain state across runs, access the current contexts memory allocators functionality, and interact with the current contexts event system. For the most part, the boiler plate bits will be handled automatically by the code generation tools, but if you really need it for things like quick profiling then it is pretty easy to extend!

The two allocators are of particular importance here as they help guarantee that uTensor is dynamically isolated from the rest of the system code:

- **Metadata Allocator**: When `new` or `delete` is called for tensor, all of this tensors object data gets allocated, constructed, and  placed in the currently set Metadata Allocator. Things like dynamically constructed events and even operator objects can go here too! The best way to think about this operator is "objects go in the metadata allocator". Consequently, be sure to construct and register this allocator before constructing any tensors, otherwise the tensor will be out of context!
- **RAM data Allocator**: When the user or uTensor runtime needs a large chunk of temporary storage which guarantees the uTensor contractual behavior they can request it via this allocator. For example, the `RamTensor` basically just contains a `Handle` bound to a region in this allocator, and the read/write interface simply interacts with this buffer. This allocator is also super useful for operators that need to request a temporary scratch buffer for evaluation. The best way to think about this allocator is "temporary data blobs go in the RAM data allocator". 

### Operators

The operators are grouped by functionality and are **tensor agnostic**, that is all the inputs and outputs to `Operators` are just `Handle`s with syntactic sugar called `Tensor`s bound to objects that implement the `TensorInterface` like `RomTensor` or `RamTensor`:

- legacy: contains the old asymmetric quantized reference operators which have high RAM requirements in practice.
- optimized: contains optimized forms of the reference operators, for example with Arm targets this will use the CMSIS-NN and CMSIS-DSP optimized libraries under the hood.
- symmetric_quantization: Reference operators based on the relatively new standard symmetric quantization schemes.
- reference operators are contained in the root of the operator directory and is meant as an easy to read/understand version of the operators. These should not be used in performance critical applications.

uTensor aims to be a strong research vehicle which enables quick development cycles in the various components. Consequently, we adopt an interoperability specification with Googles Tensorflow Lite Micro's operators; our operators provide the same inputs, outputs, parameters, and generate bit accurate results against the various TLFu operators. This means we can quickly transfer learnings from one system to another with minimal friction, and without forcing our users to think about tiny differences in behavior between frameworks. 

General ops must use the tensor read/write interface for clarity, which does have a small performance hit. Optimized operators are signified by inheriting the FastOperator label class. These ops have the ability to read/write directly to blocks of memory. This way operator performance is slightly decoupled and is a combination of Tensor R/W performance and Operator throughput. 

Generically operators look like the following:

```cpp
class MyOperator : public OperatorInterface<num_inputs, num_outputs> {
 public:
  enum names_in : uint8_t { a, b }; // these are named IDs for the inputs
  enum names_out : uint8_t { c };   // these are named IDs for the outputs

  // Optional constructor

 protected:
  virtual void compute() {
    // Operator interface maintains a 2 maps of tensor names to tensors, one for inputs, and one for outputs 
    my_kernel(outputs[c].tensor(), inputs[a].tensor(), inputs[b].tensor());
  }
};

MyOperator myOp;
myOp
  .setInputs({{myOp::a, tensor1}, {myOp::b, tensor2}}) //Bind tensors to input names in op
  .setOutputs({{myOp::c, tensor3}})                    //Bind tensors to output names in op
  .eval();                                             //Evaluate operator given inputs/outputs
```

Note the tensors are referenced by these keys, so order does not matter. The following is equivalent.

```cpp
MyOperator myOp;
myOp
  .setInputs({{myOp::b, tensor2}, {myOp::a, tensor1}}) //Bind tensors to input names in op
  .setOutputs({{myOp::c, tensor3}})                    //Bind tensors to output names in op
  .eval();                                             //Evaluate operator given inputs/outputs
```


#### Table of Operators

Fully supported operators:

| Operator                    | Optimized Form (Y/N)  | Internal Temporary Buffer Allocation |
| --------------------------- | --------------------- | ------------------------------------ |
| ReLU                        | N                     | 0                                    |
| ReLU In place               | N                     | 0                                    |
| ReLU6                       | N                     | 0                                    |
| ReLU6 In place              | N                     | 0                                    |
| ArgMax                      | N                     | 0                                    |
| ArgMin                      | N                     | 0                                    |
| AddOperator                 | N                     | 0                                    |
| Conv2D                      | N                     | 0                                    |
| QuantizedConv2D             | N                     | 0                                    |
| OptConv2D                   | Y                     | 4*ker_x*ker_y*in_channels            |
| MinPool                     | N                     | 0                                    |
| MaxPool                     | N                     | 0                                    |
| AvgPool                     | N                     | 0                                    |
| GenericPool                 | N                     | 0                                    |
| DepthwiseSepConv2D          | N                     | 0                                    |
| QuantizedDepthwiseSepConv2D | N                     | 0                                    |
| OptDepthwiseSepConv2D       | Y                     | 0                                    |
| Min                         | N                     | 0                                    |
| Max                         | N                     | 0                                    |
| Squeeze                     | N                     | 0                                    |
| MatMul                      | N                     | 0                                    |
| QuantizedFullyConnected     | N                     | 0                                    |
| Reshape                     | N                     | 0                                    |

This list is not complete, as there are a handful of operators that are in the repo, but not fully tested. Note, this does table also does not include legacy operators.

#### References:

- [Quantization whitepaper](https://arxiv.org/pdf/1806.08342.pdf)

### Tensors

The various `TensorInterface` implementations are the bread and butter of uTensor and describe where and how the underlying data is accessed. For example, `RomTensors` are read-only tensors which access data in ROM. `BufferTensors` are a R/W generalization of `RomTensor` where the user space or application space owns and allocates the buffer of the underlying data. RamTensors are temporary tensors that have data allocated in and bound to the `RamAllocator`, and when they go out of scope this data is automatically freed. 

There are many potentially useful tensors to solve a variety of tasks not included in the default. For example, you could implement a `OffChipRamTensor`, `NPUTensor`, `DMATensor`, `CameraTensor`, `TimeSeriesTensor`, and many more. There is nothing prohibiting tensors from referencing data that isnt even on the main chip!

#### Tensor Read Write interface

For performance reasons, the various Tensors read/write interfaces behave more like buffers than full-fledged C++ typed objects, even though the high level interface looks very Pythonic in nature. The actual reading and writing depends on how the user casts this buffer, for example:

```cpp
uint8_t myBuffer[4] = { 0xde, 0xad, 0xbe, 0xef };
Tensor mTensor = new BufferTensor({2,2}, u8, myBuffer); // define a 2x2 tensor of uint8_ts

uint8_t a1 = mTensor(0,0);  // implicitly casts the memory referenced at this index to a uint8_t
printf("0x%hhx\n", a1);     // prints 0xde

uint16_t a2 = mTensor(0,0); // implicitly casts the memory referenced at this index to a uint16_t
printf("0x%hx\n", a2);      // prints 0xdead

uint32_t a3 = mTensor(0,0); // implicitly casts the memory referenced at this index to a uint32_t
printf("0x%x\n", a3);      // prints 0xdeadbeef

// You can also write and read values with explicit casting and get similar behavior
mTensor(0,0) = static_cast<uint8_t>(0xFF);
printf("0xhhx\n", static_cast<uint8_t>(mTensor(0,0)));
```

Although weird, this is actually the intended behavior since it allows us to fetch multiple values at once and possibly gain some acceleration from math intrinsics without sacrificing the readablitiy/understandability of the code. The only things allowed to access big blocks of data directly are those given expressed permission to do so, like the FastOperators. 
