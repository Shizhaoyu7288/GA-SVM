% model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');
% [predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model);
% template.x = [];
% template.y = [];
model = svmtrain(data_label,data_train , '-c 1 -g 0.07');
[predict_label, accuracy, dec_values] = svmpredict(data_label,data_train , model);
CR = ClassResult(data_label,data_train,model);
