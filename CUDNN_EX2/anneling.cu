learning_rate = static_cast<float>(learning_rate * pow(10, -(iter / 2))); // 2epoch �� 0.1 �� �� ��ܽ� ���� (step decay):
																		  //learning_rate = static_cast<float>(learning_rate * exp((-1)*iter*0.001)); // ������ ���� (exponential decay)
																		  //learning_rate = static_cast<float>(learning_rate /(1+iter*0.001)); // 1/t ����