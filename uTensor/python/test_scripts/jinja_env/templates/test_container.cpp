{# This jinja2 template are designed for testing single op outputs#}
#include <cstring>
#include <iostream>

#include "arenaAllocator.hpp"
#include "uTensor/core/context.hpp"
{%for header in utensor_headers%}
#include "{{header}}"
{%endfor%}

#include "gtest/gtest.h"

{%for header in test_headers%}
#include "{{header}}"
{%endfor%}

using namespace uTensor;
 
TEST({{test_suit_name}}, {{test_name}}) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<{{ output_size }}*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  {%for declare in declare_tensor_strs%}
  {{declare}}
  {%endfor%}
  {%if op_construct_params%}
  {{op_cls}}<{{op_type_signature}}> {{op_name}}({%for param in op_construct_params%}{{param}}{{"," if not loop.last}}{%endfor%});
  {%else%}
  {{op_cls}}<{{op_type_signature}}> {{op_name}};
  {%endif%}
  {{op_name}}
    .set_inputs({ {{inputs_str}} })
    .set_outputs({ {{outputs_str}} })
    .eval();
  {%for output_name, ref_output_name in zip(output_names, ref_output_names)%}
  for (int i = 0; i < {{output_size}}; ++i) {
    {{output_type_str}} value = static_cast<{{output_type_str}}>({{output_name}}(i));
    {%if tol %}
    EXPECT_NEAR(value, {{ref_output_name}}[i], {{tol}});
    {%else%}
    EXPECT_EQ(value, {{ref_output_name}}[i]);
    {%endif%}
  }
  {%endfor%}
  {%if other_tests_str%}
  {%for test_template_str in other_tests_str%}
  {{test_template_str}}
  {%endfor%}
  {%endif%}
}