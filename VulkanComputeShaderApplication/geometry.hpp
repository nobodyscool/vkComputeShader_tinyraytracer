#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
struct Material
{
	glm::vec4 albedo = { 2,0,0,0 };
	glm::vec4 diffuse_color_specular_exponent = { 0,0,0,0 };
	glm::vec4 refractive_index = { 1.0, 0, 0, 0 };
};

struct Sphere
{
	glm::vec4 center_radius;
	Material material;
};

struct Triangle
{
	glm::vec4 v0;
	glm::vec4 v1;
	glm::vec4 v2;
	Material material;
};

struct Model
{	// 为了满足std140 布局的对齐规则将前三个参数整合到params0中
	glm::vec4 params0; // x=startIndex, y=count, z=normalinterpolatio (0/1), w=triangleCount
	//int startIndex:起始索引
	//int count:三角形数量
	//bool normalinterpolatio:是否启动表面平滑
	//int triangleCount:三角形总数
	glm::vec4 bboxMin;
	glm::vec4 bboxMax;
	Material material;

	//构造函数
	Model(const Material& mat)
		: params0(0,0,0,0),
		bboxMin(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX),
		bboxMax(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
		material(mat) {}
};