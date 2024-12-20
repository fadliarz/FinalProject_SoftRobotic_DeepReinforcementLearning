clc
clear

addpath(genpath('..\Toolboxes\SoRoRim\Basic functions'))
addpath('..\Toolboxes\SoRoRim\Custom')
addpath('..\Toolboxes\SoRoRim\SorosimLink files')
addpath('..\Toolboxes\SoRoRim\SorosimTwist files')
addpath(genpath('..\Toolboxes\SoRoRim\SorosimLinkage files'))

link1 = SorosimLink;

save(fullfile('C:\Users\Fadli\Desktop\workspace\FinalProject_SoftRobotic_DeepReinforcementLearning\src\matlab\src\generate\mat', 'link1.mat'), 'link1')