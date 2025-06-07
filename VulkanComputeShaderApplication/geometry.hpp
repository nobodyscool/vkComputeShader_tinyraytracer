#pragma once
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <string>
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

// ��ɫ����ʹ�õ�ģ������
struct Model
{	// Ϊ������std140 ���ֵĶ������ǰ�����������ϵ�params0��
	glm::ivec4 params0; // x=startIndex, y=count, z=normalinterpolatio (0/1), w=padding
	//int startIndex:��ʼ����
	//int count:����������
	//bool normalinterpolatio:�Ƿ���������ƽ��
	glm::vec4 bboxMin;
	glm::vec4 bboxMax;
	Material material;

	//���캯��
	Model(const Material& mat)
		: params0(0,0,0,0),
		bboxMin(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX),
		bboxMax(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
		material(mat) {}
};

// ����ģ��ʱʹ�õ���Ϣ�ṹ
struct ModelInfo
{	
	Model model;
	std::string filename;
	glm::vec3 scale;
	glm::vec3 rotation;
	glm::vec3 translation;

	ModelInfo(
		const Model& model_,
		const std::string& filename_,
		const glm::vec3& scale_,
		const glm::vec3& rotation_,
		const glm::vec3& translation_
	)
		: model(model_)
		, filename(filename_)
		, scale(scale_)
		, rotation(rotation_)
		, translation(translation_)
	{}
};