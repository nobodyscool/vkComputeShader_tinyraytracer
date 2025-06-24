#pragma once
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include "geometry.hpp"
// 通用

// Duck 配置
constexpr Material duckMaterial = { {0.0,0.5, 0.1, 0.8}, {0.6, 0.7, 0.8, 125.0},  {1.5, 0.0, 0.0, 0.0} };
constexpr glm::vec3 duckScale = { 1.0f, 1.0f, 1.0f };
constexpr glm::vec3 duckRotation = { 0.0f, 0.0f, 0.0f };
constexpr glm::vec3 duckTranslation = { 0.0f, 0.0f, 0.0f };
constexpr int duckNormalinterpolation = 1;
Model Duck(duckMaterial, duckNormalinterpolation);
ModelInfo duckInfo(Duck, "assets/duck.obj", duckScale, duckRotation, duckTranslation);


// Asschercut 配置
constexpr Material asschercutMaterial = { {0.0f, 0.5f, 0.1f, 0.8f}, {0.6f, 0.7f, 0.8f, 125.0f}, {1.5f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 asschercutScale = { 2.0f, 2.0f, 2.0f };
constexpr glm::vec3 asschercutRotation = { 45.0f, 45.0f, 0.0f };
constexpr glm::vec3 asschercutTranslation = { 0.2f, -2.0f, -14.0f };
constexpr int asschercutNormalinterpolation = 0;
Model Asschercut(asschercutMaterial, asschercutNormalinterpolation);
ModelInfo asschercutInfo(Asschercut, "assets/asschercut-mesh.obj", asschercutScale, asschercutRotation, asschercutTranslation);

// Bunny 配置
constexpr Material bunnyMaterial = { {0.9f, 0.5f, 0.0f, 0.0f}, {0.0522876f, 0.156863f, 0.117647f, 10.0f}, {1.0f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 bunnyScale = { 1.0f,  1.0f,  1.0f };
constexpr glm::vec3 bunnyRotation = { 0.0f, 0.0f, 0.0f };
constexpr glm::vec3 bunnyTranslation = { 2.5f, -1.5f, -5.5f };
constexpr int bunnyNormalinterpolation = 0;
Model Bunny(bunnyMaterial, bunnyNormalinterpolation);
ModelInfo bunnyInfo(Bunny, "assets/bunny-mesh.obj", bunnyScale, bunnyRotation, bunnyTranslation);

// Dragon 配置
constexpr Material dragonMaterial = { {0.9f, 0.1f, 0.0f, 0.0f}, {0.013072f, 0.078431f, 0.143791f, 10.0f}, {1.0f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 dragonScale = { 0.7f, 0.7f, 0.7f };
constexpr glm::vec3 dragonRotation = { 0.0f, 110.0f, 0.0f };
//<Translation x = "-4.15" y = "-3.0" z = "-7.0" / >
constexpr glm::vec3 dragonTranslation = { -2.5f, -2.175f, -4.45f };
constexpr int dragonNormalinterpolation = 1;
Model Dragon(dragonMaterial, dragonNormalinterpolation);
ModelInfo dragonInfo(Dragon, "assets/dragon-mesh.obj", dragonScale, dragonRotation, dragonTranslation);

// Venus 配置
constexpr Material venusMaterial = { {0.0, 10.0, 0.8, 0.0}, {1.0, 1.0, 1.0, 1425.0}, {1.0, 0.0, 0.0, 0.0} };
constexpr glm::vec3 venusScale = { 5.5f, 5.5f, 5.5f };
constexpr glm::vec3 venusRotation = { 0.0f, 0.0f, 0.0f };
constexpr glm::vec3 venusTranslation = { -1.0f, 2.0f, -19.0f };
constexpr int venusNormalinterpolation = 1;
Model Venus(venusMaterial, venusNormalinterpolation);
ModelInfo venusInfo(Venus, "assets/venus-mesh.obj", venusScale, venusRotation, venusTranslation);

// FudanLogo 配置
constexpr Material fudanlogoMaterial = { {0.7f, 0.7f, 0.0f, 0.0f}, { 0.515625f, 0.3984275f, 0.32421875f, 12.0f }, {1.0f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 fudanlogoScale = { 0.4f, 0.4f, 0.4f };
constexpr glm::vec3 fudanlogoRotation = { 90.0f, 0.0f, 0.0f }; 
constexpr glm::vec3 fudanlogoTranslation = { 0.85f, 1.95f, -4.15f };
constexpr int fudanLogoNormalinterpolation = 0;
Model FudanLogo(fudanlogoMaterial, fudanLogoNormalinterpolation);
ModelInfo fudanlogoInfo(FudanLogo, "assets/fudanlogo-mesh.obj", fudanlogoScale, fudanlogoRotation, fudanlogoTranslation);

// 导入模型的列表
std::vector<ModelInfo> modelList = {asschercutInfo ,bunnyInfo ,dragonInfo,venusInfo ,fudanlogoInfo };
//std::vector<ModelInfo> modelList = { asschercutInfo };