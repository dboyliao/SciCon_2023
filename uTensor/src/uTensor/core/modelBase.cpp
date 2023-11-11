#include "modelBase.hpp"

#include "uTensor/core/context.hpp"

namespace uTensor {

// ModelBase::ModelBase(TensorMapInterface* inputs, TensorMapInterface* outputs)
// : inputs(inputs), outputs(outputs) {
//    Context* ctx = Context::get_default_context();
//    //        ctx.push_op_tensors(*this, inputs);
//    //        ctx.push_op_tensors(*this, outputs);
//}
// The preferred interface
ModelBase::ModelBase() {}
ModelBase::ModelBase(uTensor::string _name) : _name(_name) {}
void ModelBase::set_name(uTensor::string _name) { this->_name = _name; }

ModelBase::~ModelBase() {
  // Context* ctx = Context::get_default_context();
  // ctx.pop_op_tensors(*this, inputs); // Inputs are no longer needed
}
void ModelBase::eval() { compute(); }

ModelBase::ModelBase(TensorMapInterface* inputs) : _p_inputs(inputs) {
  // Context* ctx = Context::get_default_context();
}
void ModelBase::set_inputs(TensorMapInterface* inputs) {
  this->_p_inputs = inputs;
  // Context* ctx = Context::get_default_context();
  // ctx.push_op_tensors(*this, inputs);
}
void ModelBase::set_outputs(TensorMapInterface* outputs) {
  this->_p_outputs = outputs;
  // Context* ctx = Context::get_default_context();
  // ctx.push_op_tensors(*this, outputs);
}

}  // namespace uTensor
