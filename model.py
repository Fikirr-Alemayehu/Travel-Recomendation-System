# ===== BUILDSVDMODEL =====

# Train the SVD model
#model = SVD()
model = SVD(random_state=SEED)
model.fit(trainset)

# Evaluate model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print("RMSE:", rmse)

