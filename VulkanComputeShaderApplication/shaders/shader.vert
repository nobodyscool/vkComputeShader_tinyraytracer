//#version 450
//#extension GL_KHR_vulkan_glsl :enable
//
//vec2 positions[3] = vec2[](
//    vec2(0.0, -0.5),
//    vec2(0.5, 0.5),
//    vec2(-0.5, 0.5)
//);
//
//layout(location = 0) out vec2 uv;
//void main() {
//    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
//    uv = (gl_Position.xy + 1.0) * 0.5;
//}

#version 450
#extension GL_KHR_vulkan_glsl :enable

vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    
    vec2(-1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0)
);

layout(location = 0) out vec2 uv;

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = pos * 0.5 + 0.5;
}