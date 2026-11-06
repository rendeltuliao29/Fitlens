package com.example.fitlensartest

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class ShoulderOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val shoulderPaintAligned = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#34D399") // Emerald green
        strokeWidth = 7f
        style = Paint.Style.STROKE
    }

    private val shoulderPaintWarning = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F59E0B") // Amber warning
        strokeWidth = 7f
        style = Paint.Style.STROKE
        pathEffect = DashPathEffect(floatArrayOf(12f, 12f), 0f)
    }

    private val chestPaintAligned = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#60A5FA") // Bright blue
        strokeWidth = 6f
        style = Paint.Style.STROKE
        pathEffect = DashPathEffect(floatArrayOf(16f, 10f), 0f)
    }

    private val jointPaintAligned = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#34D399")
        style = Paint.Style.FILL
    }

    private val jointPaintWarning = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F59E0B")
        style = Paint.Style.FILL
    }

    private val chestJointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#60A5FA")
        style = Paint.Style.FILL
    }

    private val bodyLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(128, 255, 255, 255)
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }

    // Normalized landmark coordinates
    private var leftShoulderX = 0f
    private var leftShoulderY = 0f
    private var rightShoulderX = 0f
    private var rightShoulderY = 0f

    private var leftHipX = 0f
    private var leftHipY = 0f
    private var rightHipX = 0f
    private var rightHipY = 0f

    private var imageWidth = 0
    private var imageHeight = 0

    private var hasPose = false
    private var isAligned = true

    fun updateBody(
        leftShoulderX: Float,
        leftShoulderY: Float,
        rightShoulderX: Float,
        rightShoulderY: Float,
        leftHipX: Float,
        leftHipY: Float,
        rightHipX: Float,
        rightHipY: Float,
        imageWidth: Int,
        imageHeight: Int,
        isAligned: Boolean = true
    ) {
        this.leftShoulderX = leftShoulderX
        this.leftShoulderY = leftShoulderY
        this.rightShoulderX = rightShoulderX
        this.rightShoulderY = rightShoulderY

        this.leftHipX = leftHipX
        this.leftHipY = leftHipY
        this.rightHipX = rightHipX
        this.rightHipY = rightHipY

        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        this.isAligned = isAligned

        hasPose = true
        postInvalidate()
    }

    fun clear() {
        hasPose = false
        postInvalidate()
    }

    // Map camera coordinates (normalized 0..1) to view coordinates with mirroring for front camera
    private fun mapX(x: Float, scale: Float, offsetX: Float): Float {
        return offsetX + (1f - x) * imageWidth * scale
    }

    private fun mapY(y: Float, scale: Float, offsetY: Float): Float {
        return offsetY + y * imageHeight * scale
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (!hasPose || imageWidth == 0 || imageHeight == 0) {
            return
        }

        val scale = maxOf(
            width.toFloat() / imageWidth,
            height.toFloat() / imageHeight
        )

        val scaledWidth = imageWidth * scale
        val scaledHeight = imageHeight * scale

        val offsetX = (width - scaledWidth) / 2f
        val offsetY = (height - scaledHeight) / 2f

        // Map shoulder coordinates
        val leftSX = mapX(leftShoulderX, scale, offsetX)
        val leftSY = mapY(leftShoulderY, scale, offsetY)

        val rightSX = mapX(rightShoulderX, scale, offsetX)
        val rightSY = mapY(rightShoulderY, scale, offsetY)

        // Map hip coordinates
        val leftHX = mapX(leftHipX, scale, offsetX)
        val leftHY = mapY(leftHipY, scale, offsetY)

        val rightHX = mapX(rightHipX, scale, offsetX)
        val rightHY = mapY(rightHipY, scale, offsetY)

        // Chest level is approximately 28% down from shoulder line to hip line
        val chestRatio = 0.28f
        val leftChestX = leftSX + (leftHX - leftSX) * chestRatio
        val leftChestY = leftSY + (leftHY - leftSY) * chestRatio

        val rightChestX = rightSX + (rightHX - rightSX) * chestRatio
        val rightChestY = rightSY + (rightHY - rightSY) * chestRatio

        // Draw body frame guide lines (shoulders to hips)
        canvas.drawLine(leftSX, leftSY, leftHX, leftHY, bodyLinePaint)
        canvas.drawLine(rightSX, rightSY, rightHX, rightHY, bodyLinePaint)

        // Select paint based on body alignment status
        val currentShoulderPaint = if (isAligned) shoulderPaintAligned else shoulderPaintWarning
        val currentJointPaint = if (isAligned) jointPaintAligned else jointPaintWarning

        // Draw Shoulder Measurement Line
        canvas.drawLine(leftSX, leftSY, rightSX, rightSY, currentShoulderPaint)

        // Draw Chest Measurement Line (dashed)
        if (isAligned) {
            canvas.drawLine(leftChestX, leftChestY, rightChestX, rightChestY, chestPaintAligned)
            canvas.drawCircle(leftChestX, leftChestY, 12f, chestJointPaint)
            canvas.drawCircle(rightChestX, rightChestY, 12f, chestJointPaint)
        }

        // Draw Shoulder Joint Markers
        canvas.drawCircle(leftSX, leftSY, 14f, currentJointPaint)
        canvas.drawCircle(rightSX, rightSY, 14f, currentJointPaint)
    }
}
