learning_rate = static_cast<float>(learning_rate * pow(10, -(iter / 2))); // 2epoch 당 0.1 배 씩 계단식 감소 (step decay):
																		  //learning_rate = static_cast<float>(learning_rate * exp((-1)*iter*0.001)); // 지수적 감소 (exponential decay)
																		  //learning_rate = static_cast<float>(learning_rate /(1+iter*0.001)); // 1/t 감소