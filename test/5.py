# 编译
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),  # Adam优化器
              # optimizer=optimizers.RMSprop(learning_rate=0.0001),  # rmsprop优化器
              metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,  # 生成训练集生成器
    steps_per_epoch=243,  # train_num/batch_size=128
    epochs=40,  # 数据迭代轮数
    validation_data=validation_generator,  # 生成验证集生成器
    validation_steps=28  # valid_num/batch_size=128
)

# 评估模型
test_loss, test_acc = model.evaluate(test_generator, steps=28)
print("test_loss: %.4f - test_acc: %.4f" % (test_loss, test_acc * 100))

# 保存模型
model_json = model.to_json()
with open('myModel_2_json.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('myModel_2_weight.h5')
model.save('myModel_2.h5')

with open('fit_2_log.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt, 0)