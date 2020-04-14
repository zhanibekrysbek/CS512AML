function Q3_Q4_plot_figures
clear
clc
close all
% 第一列是c值，第二列是word-wise-accuracy，第三列是letter-wise-accuracy
crf = [0.00100000 0.0 0.30277120390869533;
0.01000000 0.0 0.3057867012749065;
0.10000000 0.0 0.33834643865943964;
1.00000000 0.0119220703692934 0.4483166653943049;
10.00000000 0.16196568769991276 0.691541339033514;
100.00000000 0.38092468740913055 0.8055576761584854;
1000.00000000 0.47252108170979934 0.8375448507519658;
10000.00000000 0.4870601919162547 0.8428887701351249
];
baseline_liblinear = [0.00100000 0.00000000 0.30265669;
0.01000000 0.00000000 0.32032980;
0.10000000 0.00029078 0.38548744;
1.00000000 0.02006397 0.48362470;
10.00000000 0.07647572 0.61184060;
100.00000000 0.14917127 0.68050996;
1000.00000000 0.16807211 0.69726697;
10000.00000000 0.17185228 0.69967173
];
baseline_svmhmm = [0.00100000 0.00000000 0.24776701;
0.01000000 0.00290782 0.35109550;
0.10000000 0.08112823 0.60138178;
1.00000000 0.13405060 0.65222536;
10.00000000 0.26490259 0.75043896;
100.00000000 0.41436464 0.82345981;
1000.00000000 0.47746438 0.84754561;
10000.00000000 0.49229427 0.85151538
];

% (3a) figure 1: Letter wise accuracy using CRF, SVM-MC and SVM-Struct
C = baseline_liblinear(:,1);
accuracy_CRF = crf(:,3);
accuracy_SVM_MC = baseline_liblinear(:,3);
accuracy_SVM_Struct = baseline_svmhmm(:,3);
figure(1);
semilogx(C,accuracy_CRF,'r--*',C,accuracy_SVM_MC,'b--o',C,accuracy_SVM_Struct,'g--d','LineWidth',2);
xlabel('C','fontsize',16);set(gca, 'LineWidth',3)
ylabel('Accuracy','fontsize',16);set(gca, 'LineWidth',3)
title('Letter wise accuracy using CRF, SVM-MC and SVM-Struct','fontsize',16);set(gca, 'LineWidth',3)
grid on;
h = legend('CRF',...
       'SVM-MC', ...
       'SVM-Struct', ...
       'Location', 'BestOutside');
set(h,'Fontsize',16);

% (3b) figure 2: Word wise accuracy using CRF, SVM-MC, and SVM-Struct
C = baseline_liblinear(:,1);
accuracy_CRF = crf(:,2);
accuracy_SVM_MC = baseline_liblinear(:,2);
accuracy_SVM_Struct = baseline_svmhmm(:,2);
figure(2);
semilogx(C,accuracy_CRF,'r--*',C,accuracy_SVM_MC,'b--o',C,accuracy_SVM_Struct,'g--d','LineWidth',2);
xlabel('C','fontsize',16);set(gca, 'LineWidth',3)
ylabel('Accuracy','fontsize',16);set(gca, 'LineWidth',3)
title('Word wise accuracy using CRF, SVM-MC and SVM-Struct','fontsize',16);set(gca, 'LineWidth',3)
grid on;
h = legend('CRF',...
       'SVM-MC', ...
       'SVM-Struct', ...
       'Location', 'BestOutside');
set(h,'Fontsize',16);

baseline_liblinear_x0 = [0.00100000 0.00000000 0.30265669;
0.01000000 0.00000000 0.32032980;
0.10000000 0.00029078 0.38548744;
1.00000000 0.02006397 0.48362470;
10.00000000 0.07647572 0.61184060;
100.00000000 0.14917127 0.68050996;
1000.00000000 0.16807211 0.69726697;
10000.00000000 0.17185228 0.69967173
];
baseline_liblinear_x500 = [0.00100000 0.00000000 0.27883808;
0.01000000 0.00000000 0.30479426;
0.10000000 0.00000000 0.37869303;
1.00000000 0.01424833 0.47037942;
10.00000000 0.06571678 0.59172456;
100.00000000 0.12503635 0.65459195;
1000.00000000 0.13288747 0.66638675;
10000.00000000 0.13463216 0.66787541
];
baseline_liblinear_x1000 = [0.00100000 0.00000000 0.26914268;
0.01000000 0.00000000 0.28750286;
0.10000000 0.00000000 0.37060081;
1.00000000 0.01163129 0.45614169;
10.00000000 0.05437627 0.57565463;
100.00000000 0.10613550 0.63798763;
1000.00000000 0.11369584 0.64775937;
10000.00000000 0.11427741 0.64852279
];
baseline_liblinear_x1500 = [0.00100000 0.00000000 0.26040156;
0.01000000 0.00000000 0.27712039;
0.10000000 0.00000000 0.36014200;
1.00000000 0.00872347 0.44106420;
10.00000000 0.04390811 0.55611115;
100.00000000 0.08839779 0.62058172;
1000.00000000 0.10148299 0.62981907;
10000.00000000 0.10293690 0.63107871
];
baseline_liblinear_x2000 = [0.00100000 0.00000000 0.23956027;
0.01000000 0.00000000 0.25322544;
0.10000000 0.00000000 0.34094206;
1.00000000 0.00756034 0.42117719;
10.00000000 0.03692934 0.53252157;
100.00000000 0.07531259 0.59458737;
1000.00000000 0.09014248 0.60504619;
10000.00000000 0.08927014 0.60462631
];

c100_crf_letter_wise = [0 0.8055576761584854;
500 0.7833422398656386;
1000 0.7703259790823727;
1500 0.7516986029467898;
2000 0.7209710664936254
];

c100_crf_word_wise = [0 0.38092468740913055;
500 0.344286129688863;
1000 0.3146263448676941;
1500 0.28642047106717067;
2000 0.2471648735097412
];

% (4a) figure 3: Letter wise accuracy using SVM-MC and CRF
C = [0, 500, 1000, 1500, 2000];
accuracy_CRF = c100_crf_letter_wise(:,2);
num_c = 8; % plot results of Q4 with C=10000, which achieve best accuracy in Q3(a)
accuracy_SVM_MC = [baseline_liblinear_x0(num_c,3), baseline_liblinear_x500(num_c,3), baseline_liblinear_x1000(num_c,3), baseline_liblinear_x1500(num_c,3), baseline_liblinear_x2000(num_c,3)];
figure(3);
plot(C,accuracy_CRF,'r--*',C,accuracy_SVM_MC,'b--o','LineWidth',2);
xlabel('Number of words tampered','fontsize',16);set(gca, 'LineWidth',3)
ylabel('Accuracy','fontsize',16);set(gca, 'LineWidth',3)
title('Letter wise accuracy using CRF, SVM-MC','fontsize',16);set(gca, 'LineWidth',3)
grid on;
h = legend('CRF',...
       'SVM-MC', ...
       'Location', 'BestOutside');
set(h,'Fontsize',16);

% (4b) figure 2: Word wise accuracy using CRF and SVM-MC
C = [0, 500, 1000, 1500, 2000];
accuracy_CRF = c100_crf_word_wise(:,2);
accuracy_SVM_MC = [baseline_liblinear_x0(num_c,2), baseline_liblinear_x500(num_c,2), baseline_liblinear_x1000(num_c,2), baseline_liblinear_x1500(num_c,2), baseline_liblinear_x2000(num_c,2)];
figure(4);
plot(C,accuracy_CRF,'r--*',C,accuracy_SVM_MC,'b--o','LineWidth',2);
xlabel('Number of words tampered','fontsize',16);set(gca, 'LineWidth',3)
ylabel('Accuracy','fontsize',16);set(gca, 'LineWidth',3)
title('Word wise accuracy using CRF, SVM-MC','fontsize',16);set(gca, 'LineWidth',3)
grid on;
h = legend('CRF',...
       'SVM-MC', ...
       'Location', 'BestOutside');
set(h,'Fontsize',16);
end

